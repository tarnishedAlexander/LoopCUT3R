#!/usr/bin/env python3
"""
stage_to_nvme.py — NVMe Dataset Staging for LoopCUT3R
======================================================
Copies HDF5 dataset files from HDD archive to NVMe SSD, then replaces the
original HDD file with a symlink pointing to the NVMe copy. This makes the
staging completely transparent to the existing H5VFSRegistry — no config
changes required, no dataset classes need modification.

Why symlinks instead of config edits:
  The H5VFSRegistry scans subdirectories in os.listdir() order (non-
  deterministic). A symlink at the original path guarantees the VFS always
  opens the NVMe copy, regardless of scan order.

Priority ordering:
  Files are staged in descending samples-per-GB order — maximizing the
  fraction of training samples served from NVMe per unit of NVMe space used.

Usage:
  python stage_to_nvme.py                    # stage all that fit
  python stage_to_nvme.py --dry-run          # show plan without copying
  python stage_to_nvme.py --unstage          # revert all symlinks to HDD
  python stage_to_nvme.py --status           # show current staging state
"""

import os
import sys
import shutil
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE = Path(os.environ.get("LOOPCUT3R_BASE", "/app/recurrentTransformer"))
NVME = BASE / "dataset/datasetfaster"
D12  = BASE / "dataset/disk12tb"
D14  = BASE / "dataset/disk14tb"
POOL = BASE / "dataset/diskPool"

# ── Staging priority table ─────────────────────────────────────────────────────
# Ordered by (stage2_samples / file_size_gb) — most efficient use of NVMe space.
# source: path to the real HDF5 file on HDD
# stage2_samples: number of samples drawn per epoch in recurrent_stage2

@dataclass
class DatasetEntry:
    name: str
    source: Path          # actual HDD .h5 file
    stage2_samples: int   # samples drawn per epoch in stage2 config

DATASETS = [
    DatasetEntry("MapFree",      D14  / "processed_mapfree.h5",          60_000),
    DatasetEntry("WildRGBD",     D14  / "preprocess_wildrgbd.h5",        60_000),
    DatasetEntry("Co3D",         D14  / "processed_co3d.h5",             48_000),
    DatasetEntry("MegaDepth",    D14  / "processed_megadepth.h5",         6_000),
    DatasetEntry("ScanNetpp",    D14  / "processed_scannetpp.h5",         6_000),
    DatasetEntry("BlendedMVS1",  D14  / "processed_BlendedMVS1_full.h5",  6_000),
]

NVME_SAFETY_MARGIN_GB = 20.0   # leave this much free on NVMe at all times


# ── Helpers ────────────────────────────────────────────────────────────────────

def gb(n_bytes: int) -> float:
    return n_bytes / 1024**3


def nvme_free_gb() -> float:
    stat = shutil.disk_usage(NVME)
    return gb(stat.free)


def file_size_gb(path: Path) -> Optional[float]:
    """Returns size in GB, or None if path doesn't exist / is a broken symlink."""
    try:
        return gb(path.stat().st_size)
    except (FileNotFoundError, OSError):
        return None


def nvme_dest(entry: DatasetEntry) -> Path:
    return NVME / entry.source.name


def is_staged(entry: DatasetEntry) -> bool:
    """True if source path is a symlink pointing to the NVMe copy."""
    src = entry.source
    dst = nvme_dest(entry)
    return src.is_symlink() and src.resolve() == dst.resolve()


def samples_per_gb(entry: DatasetEntry) -> float:
    size = file_size_gb(entry.source)
    if not size:
        return 0.0
    return entry.stage2_samples / size


# ── Core operations ────────────────────────────────────────────────────────────

def stage_entry(entry: DatasetEntry, dry_run: bool = False) -> bool:
    """
    Copy source H5 to NVMe, then replace source with a symlink.
    Returns True on success.
    """
    src  = entry.source
    dst  = nvme_dest(entry)
    size = file_size_gb(src)

    if size is None:
        log.warning("[%s] Source not found: %s — skipping", entry.name, src)
        return False

    if is_staged(entry):
        log.info("[%s] Already staged (%.1f GB) — skipping", entry.name, size)
        return True

    free = nvme_free_gb()
    needed = size + NVME_SAFETY_MARGIN_GB
    if free < needed:
        log.warning(
            "[%s] Not enough NVMe space: need %.1f GB, have %.1f GB free — skipping",
            entry.name, needed, free,
        )
        return False

    log.info(
        "[%s] Staging %.1f GB  %s  →  %s",
        entry.name, size, src, dst,
    )

    if dry_run:
        log.info("[%s] DRY RUN — no files written", entry.name)
        return True

    # ── 1. Copy to NVMe ────────────────────────────────────────────────────────
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        shutil.copy2(str(src), str(dst))
    except Exception as exc:
        log.error("[%s] Copy failed: %s", entry.name, exc)
        dst.unlink(missing_ok=True)
        return False

    elapsed = time.monotonic() - t0
    speed   = size / elapsed if elapsed > 0 else 0
    log.info("[%s] Copied in %.0fs  (%.1f GB/s)", entry.name, elapsed, speed)

    # ── 2. Verify integrity (size match) ───────────────────────────────────────
    src_sz = src.stat().st_size
    dst_sz = dst.stat().st_size
    if src_sz != dst_sz:
        log.error(
            "[%s] Size mismatch after copy: src=%d dst=%d — aborting, removing dst",
            entry.name, src_sz, dst_sz,
        )
        dst.unlink(missing_ok=True)
        return False

    # ── 3. Swap: rename original to .hdd_backup, create symlink ───────────────
    backup = src.with_suffix(".h5.hdd_original")
    try:
        src.rename(backup)          # HDD file → .hdd_original (preserves the data)
        # Use a RELATIVE symlink so it resolves correctly both on host and
        # inside the Docker container (which mounts the parent at a different
        # absolute prefix). os.path.relpath gives e.g. "../datasetfaster/X.h5"
        rel_dst = os.path.relpath(dst, src.parent)
        src.symlink_to(rel_dst)
    except Exception as exc:
        log.error("[%s] Symlink swap failed: %s — rolling back", entry.name, exc)
        if backup.exists():
            backup.rename(src)      # restore original name
        dst.unlink(missing_ok=True)
        return False

    log.info(
        "[%s] ✓ Staged. NVMe reads transparent via symlink: %s → %s",
        entry.name, src, dst,
    )
    return True


def unstage_entry(entry: DatasetEntry, dry_run: bool = False) -> bool:
    """Revert: restore .hdd_original, remove symlink and NVMe copy."""
    src    = entry.source
    dst    = nvme_dest(entry)
    backup = src.with_suffix(".h5.hdd_original")

    if not src.is_symlink():
        log.info("[%s] Not currently staged — nothing to do", entry.name)
        return True

    if not backup.exists():
        log.error("[%s] HDD backup not found: %s — cannot unstage safely", entry.name, backup)
        return False

    log.info("[%s] Reverting symlink → HDD file", entry.name)
    if dry_run:
        return True

    try:
        src.unlink()            # remove symlink
        backup.rename(src)      # restore original HDD file
        dst.unlink(missing_ok=True)
    except Exception as exc:
        log.error("[%s] Unstage failed: %s", entry.name, exc)
        return False

    log.info("[%s] ✓ Reverted to HDD", entry.name)
    return True


# ── Status report ──────────────────────────────────────────────────────────────

def print_status():
    total_stage2_samples = 762_980  # recurrent_stage2 total
    print(f"\n{'Dataset':<16} {'Status':<12} {'Size GB':>8} {'Samples':>9} {'Samp/GB':>9}")
    print("─" * 60)

    staged_samples = 0
    staged_gb      = 0.0

    for entry in DATASETS:
        src  = entry.source
        size = file_size_gb(nvme_dest(entry) if is_staged(entry) else src)
        status = "NVMe ✓" if is_staged(entry) else (
            "HDD" if (src.exists() and not src.is_symlink()) else
            "MISSING" if not src.exists() else "BROKEN"
        )
        spg = samples_per_gb(entry) if size else 0
        print(f"{entry.name:<16} {status:<12} {(size or 0):>8.1f} {entry.stage2_samples:>9,} {spg:>9.0f}")
        if is_staged(entry):
            staged_samples += entry.stage2_samples
            staged_gb      += size or 0

    print("─" * 60)
    pct = 100 * staged_samples / total_stage2_samples
    print(f"Staged coverage: {staged_samples:,} / {total_stage2_samples:,} samples ({pct:.1f}%)")
    print(f"NVMe used by staged datasets: {staged_gb:.1f} GB")
    print(f"NVMe free: {nvme_free_gb():.1f} GB\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage LoopCUT3R HDF5 datasets to NVMe")
    parser.add_argument("--dry-run",  action="store_true", help="Show plan without writing")
    parser.add_argument("--unstage",  action="store_true", help="Revert all staged datasets")
    parser.add_argument("--status",   action="store_true", help="Show staging status")
    parser.add_argument("--only",     nargs="+", metavar="NAME",
                        help="Stage only these dataset names (e.g. --only Co3D MapFree)")
    args = parser.parse_args()

    NVME.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status()
        return

    # Sort by samples/GB descending — most efficient use of NVMe space first
    targets = sorted(DATASETS, key=samples_per_gb, reverse=True)
    if args.only:
        names  = {n.lower() for n in args.only}
        targets = [e for e in targets if e.name.lower() in names]
        if not targets:
            log.error("No matching datasets for --only %s", args.only)
            sys.exit(1)

    if args.unstage:
        for entry in targets:
            unstage_entry(entry, dry_run=args.dry_run)
        print_status()
        return

    log.info("NVMe free: %.1f GB  (safety margin: %.0f GB)", nvme_free_gb(), NVME_SAFETY_MARGIN_GB)
    success = failed = skipped = 0
    for entry in targets:
        result = stage_entry(entry, dry_run=args.dry_run)
        if result:
            success += 1
        else:
            failed += 1

    print_status()
    log.info("Done: %d staged, %d failed", success, failed)


if __name__ == "__main__":
    main()
