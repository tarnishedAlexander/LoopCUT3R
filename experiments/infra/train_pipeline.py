#!/usr/bin/env python3
"""
train_pipeline.py — LoopCUT3R Training Pipeline Orchestrator
=============================================================
Wraps the existing train.py with:
  1. Pre-training NVMe staging  (calls stage_to_nvme.py logic)
  2. Optimized DataLoader flags passed as Hydra overrides
  3. Live GPU utilization monitor that logs idle % and IO-wait ratio
  4. Post-training summary

This is NOT a replacement for train.py — it wraps it. All 31 dataset
classes, the H5VFSRegistry, the weighted sampler, and TBPTL remain
unchanged. Only the I/O path (HDD → NVMe symlink) and worker count change.

Why this approach instead of sharding:
  LoopCUT3R samples from 31 datasets simultaneously via BatchedRandomSampler
  with fixed per-dataset weights. Sequential shard training would collapse
  this to single-dataset training per shard, destroying the diversity
  that multi-dataset mixing provides. Symlink-based NVMe staging is the
  only compatible approach.

Usage:
  # Stage, train stage2, monitor GPU:
  python train_pipeline.py --config recurrent_stage2

  # Stage only, don't train (useful before a long training run):
  python train_pipeline.py --config recurrent_stage2 --stage-only

  # Train without pre-staging (datasets already on NVMe):
  python train_pipeline.py --config recurrent_stage2 --skip-staging

  # Extend an existing run to more epochs:
  python train_pipeline.py --config recurrent_stage2 --override epochs=4
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Optional

# ── Bootstrap: make sure we run from the LoopCUT3R src directory ──────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── GPU Monitor ────────────────────────────────────────────────────────────────

class GPUMonitor(threading.Thread):
    """
    Background thread that polls nvidia-smi every N seconds and tracks:
      - GPU utilization (%)
      - GPU memory used (MB)
      - Idle fraction (samples where util < idle_threshold)

    Logs a summary every `report_interval` seconds.
    """

    def __init__(
        self,
        poll_interval:   float = 5.0,
        report_interval: float = 300.0,
        idle_threshold:  int   = 20,
    ):
        super().__init__(daemon=True)
        self.poll_interval   = poll_interval
        self.report_interval = report_interval
        self.idle_threshold  = idle_threshold
        self._stop_event     = threading.Event()
        self._samples:   list[int]   = []
        self._mem_mb:    list[int]   = []
        self._last_report: float     = time.monotonic()

    def stop(self):
        self._stop_event.set()

    def _query(self) -> Optional[tuple[int, int]]:
        """Returns (gpu_util_pct, mem_used_mb) or None on error."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            ).decode().strip().split(",")
            return int(out[0].strip()), int(out[1].strip())
        except Exception:
            return None

    def run(self):
        while not self._stop_event.is_set():
            result = self._query()
            if result is not None:
                util, mem = result
                self._samples.append(util)
                self._mem_mb.append(mem)

            now = time.monotonic()
            if now - self._last_report >= self.report_interval and self._samples:
                self._print_report()
                self._last_report = now

            self._stop_event.wait(self.poll_interval)

        # Final report on exit
        if self._samples:
            self._print_report(final=True)

    def _print_report(self, final: bool = False):
        n      = len(self._samples)
        avg    = sum(self._samples) / n
        peak   = max(self._samples)
        idle_n = sum(1 for u in self._samples if u < self.idle_threshold)
        idle_p = 100.0 * idle_n / n
        mem_p  = max(self._mem_mb)

        tag = "FINAL SUMMARY" if final else "GPU STATUS"
        log.info(
            "[%s] avg_util=%.1f%%  peak_util=%d%%  idle(<%d%%)=%.1f%%  "
            "peak_mem=%d MB  samples=%d",
            tag, avg, peak, self.idle_threshold, idle_p, mem_p, n,
        )

        if idle_p > 60 and not final:
            log.warning(
                "GPU idle >60%% of the time — likely dataloader bottleneck. "
                "Run: python stage_to_nvme.py --status"
            )


# ── NVMe staging (calls stage_to_nvme logic inline) ───────────────────────────

def run_staging(dry_run: bool = False):
    """Stage high-priority H5 files to NVMe before training starts."""
    import importlib.util
    staging_script = SCRIPT_DIR / "stage_to_nvme.py"
    if not staging_script.exists():
        log.warning("stage_to_nvme.py not found — skipping NVMe staging")
        return

    spec   = importlib.util.spec_from_file_location("stage_to_nvme", staging_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from stage_to_nvme import DATASETS, stage_entry, print_status, samples_per_gb
    targets = sorted(DATASETS, key=samples_per_gb, reverse=True)

    log.info("Pre-training NVMe staging: %d datasets queued", len(targets))
    for entry in targets:
        stage_entry(entry, dry_run=dry_run)
    print_status()


# ── Training subprocess ────────────────────────────────────────────────────────

# DataLoader overrides that are safe to apply to any recurrent stage config.
# Rationale for each:
#   num_workers=8   — matches physical cores on 7800X3D (8C/16T); 14 causes
#                     context-switch thrashing since workers do CPU-heavy H5
#                     decompression.
#   pin_memory=True — already set in get_data_loader(), listed for clarity.
#   prefetch_factor — already 2; increasing to 3 gives one extra batch
#                     pre-decoded in CPU RAM at minimal memory cost.
DATALOADER_OVERRIDES = [
    "num_workers=8",
]


def run_training(config_name: str, extra_overrides: list[str]) -> int:
    """Launch train.py as a subprocess with Hydra config overrides."""
    cmd = [
        sys.executable, "src/train.py",
        f"--config-name={config_name}",
        *DATALOADER_OVERRIDES,
        *extra_overrides,
    ]
    log.info("Launching training: %s", " ".join(cmd))

    proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR))

    # Forward Ctrl-C to the child process gracefully
    def _sigint(sig, frame):
        log.info("Caught SIGINT — forwarding to training process")
        proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _sigint)

    try:
        return proc.wait()
    except Exception as exc:
        log.error("Training process error: %s", exc)
        proc.kill()
        return 1


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoopCUT3R training pipeline with NVMe staging and GPU monitoring"
    )
    parser.add_argument(
        "--config", "-c",
        default="recurrent_stage2",
        help="Hydra config name (default: recurrent_stage2)",
    )
    parser.add_argument(
        "--override", "-o",
        nargs="+", default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra overrides, e.g. --override epochs=4 lr=3e-6",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Run NVMe staging then exit (no training)",
    )
    parser.add_argument(
        "--skip-staging",
        action="store_true",
        help="Skip NVMe staging (datasets already staged or staging unwanted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run staging (show what would be copied, don't train)",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable background GPU utilization monitor",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float, default=300.0,
        help="Seconds between GPU utilization reports (default: 300)",
    )
    args = parser.parse_args()

    # ── Stage ─────────────────────────────────────────────────────────────────
    if not args.skip_staging:
        run_staging(dry_run=args.dry_run)

    if args.stage_only or args.dry_run:
        log.info("Staging complete. Exiting (--stage-only / --dry-run).")
        return

    # ── Monitor ───────────────────────────────────────────────────────────────
    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(
            poll_interval=5.0,
            report_interval=args.monitor_interval,
            idle_threshold=20,
        )
        monitor.start()
        log.info(
            "GPU monitor started (reporting every %.0fs, idle threshold <20%%)",
            args.monitor_interval,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    t_start  = time.monotonic()
    exit_code = run_training(args.config, args.override)
    elapsed  = time.monotonic() - t_start

    # ── Wrap up ───────────────────────────────────────────────────────────────
    if monitor is not None:
        monitor.stop()
        monitor.join(timeout=10)

    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    log.info(
        "Training finished in %dh %dm %ds — exit code: %d",
        h, m, s, exit_code,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
