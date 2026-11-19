#!/usr/bin/env bash
# =============================================================================
# run_training.sh — LoopCUT3R Global Training Script
# =============================================================================
# Run this FROM INSIDE the container:
#   cd /app/recurrentTransformer/LoopCUT3R
#   bash run_training.sh [OPTIONS]
#
# OPTIONS:
#   --stage N           Run only stage N (1-4). Default: 2
#   --all-stages        Run stages 2→3→4 sequentially (stage 1 assumed done)
#   --resume-extra N    Extend current stage by N more epochs
#   --skip-staging      Skip NVMe pre-staging
#   --dry-run           Show plan, do not copy or train
#   --override KEY=VAL  Pass extra Hydra overrides (repeatable)
#
# EXAMPLES:
#   bash run_training.sh --stage 2
#   bash run_training.sh --stage 2 --resume-extra 2
#   bash run_training.sh --all-stages
#   bash run_training.sh --stage 3 --skip-staging
#   bash run_training.sh --stage 2 --override lr=3e-6
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERR]${NC}   $*"; }
section() { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}"; }

# ── Paths (container-internal) ────────────────────────────────────────────────
WORK_DIR="/app/recurrentTransformer/LoopCUT3R"
CONFIG_PATH="${WORK_DIR}/config"
CKPT_DIR="${WORK_DIR}/checkpoints"

DATASET_BASE="/app/recurrentTransformer/dataset"
FAST_DIR="/app/recurrentTransformer/fastdatasets"
NVME_DIR="${DATASET_BASE}/datasetfaster"
D12="${DATASET_BASE}/disk12tb"
D14="${DATASET_BASE}/disk14tb"
POOL="${DATASET_BASE}/diskPool"

# ── Defaults ──────────────────────────────────────────────────────────────────
STAGE=2
ALL_STAGES=false
RESUME_EXTRA=0
SKIP_STAGING=false
DRY_RUN=false
EXTRA_OVERRIDES=()

# ── Optimized DataLoader overrides applied to every stage ─────────────────────
# num_workers=8 matches the 7800X3D's 8 physical cores. The default of 14
# causes context-switch thrashing because h5py decompression is CPU-bound
# (not I/O-bound) when reading from NVMe.
PERF_OVERRIDES=("num_workers=8")

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)        STAGE="${2}";          shift 2 ;;
        --all-stages)   ALL_STAGES=true;       shift   ;;
        --resume-extra) RESUME_EXTRA="${2}";   shift 2 ;;
        --skip-staging) SKIP_STAGING=true;     shift   ;;
        --dry-run)      DRY_RUN=true;          shift   ;;
        --override)     EXTRA_OVERRIDES+=("${2}"); shift 2 ;;
        *) err "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── GPU idle audit ────────────────────────────────────────────────────────────
gpu_audit() {
    section "GPU & I/O Optimisation Audit"

    # Verify TF32/benchmark flags are set in train.py source (they are set at
    # module level, so checking PyTorch defaults in a fresh interpreter is
    # misleading — PyTorch >= 1.12 ships with matmul.allow_tf32=False by default
    # but train.py overrides all three flags immediately on import).
    local train_py="${WORK_DIR}/src/train.py"
    local tf32_matmul tf32_cudnn bench_line
    tf32_matmul=$(grep -c "matmul.allow_tf32 = True"  "$train_py" 2>/dev/null || echo 0)
    tf32_cudnn=$(grep  -c "cudnn.allow_tf32 = True"   "$train_py" 2>/dev/null || echo 0)
    bench_line=$(grep  -c "cudnn.benchmark"            "$train_py" 2>/dev/null || echo 0)

    if (( tf32_matmul > 0 )); then
        ok "  train.py: cuda.matmul.allow_tf32 = True  (set at module level)"
    else
        warn "  train.py: cuda.matmul.allow_tf32 NOT found — check src/train.py"
    fi
    if (( tf32_cudnn > 0 )); then
        ok "  train.py: cudnn.allow_tf32 = True  (set at module level)"
    else
        warn "  train.py: cudnn.allow_tf32 NOT found — check src/train.py"
    fi
    if (( bench_line > 0 )); then
        ok "  train.py: cudnn.benchmark wired to config 'benchmark' key  (True in stage2)"
    else
        warn "  train.py: cudnn.benchmark not found — cuDNN won't auto-tune conv algorithms"
    fi

    # GPU utilisation snapshot
    if command -v nvidia-smi &>/dev/null; then
        local util mem
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
        info "GPU util right now: ${util}%  |  VRAM: ${mem}"
    fi

    # Workers recommendation
    local cores
    cores=$(nproc 2>/dev/null || echo "?")
    info "CPU logical cores: ${cores}  →  using num_workers=8 (physical cores)"

    # NVMe staging coverage
    local staged=0 total_samples=762980
    for f in \
        "${D14}/processed_mapfree.h5" \
        "${D14}/preprocess_wildrgbd.h5" \
        "${D14}/processed_co3d.h5" \
        "${D14}/processed_megadepth.h5" \
        "${D14}/processed_scannetpp.h5" \
        "${D14}/processed_BlendedMVS1_full.h5"
    do
        if [[ -L "$f" ]] && [[ "$(readlink -f "$f")" == "${NVME_DIR}/"* ]]; then
            staged=1
        fi
    done
    if [[ $staged -eq 0 ]]; then
        warn "Hot datasets NOT yet staged to NVMe — GPU will be I/O starved."
        warn "Run with  --stage X  (staging happens automatically before training)."
    else
        ok "Hot datasets are staged on NVMe."
    fi
}

# ── NVMe staging ──────────────────────────────────────────────────────────────
# Priority-ordered list: (source_h5, samples_in_stage2)
# Ordered by samples-per-GB so the most impactful data fills NVMe first.

stage_to_nvme() {
    section "NVMe Pre-Staging"

    [[ -d "${NVME_DIR}" ]] || { warn "NVMe mount not found: ${NVME_DIR} — skipping"; return; }

    local -A SOURCES=(
        ["${D14}/processed_mapfree.h5"]=60000
        ["${D14}/preprocess_wildrgbd.h5"]=60000
        ["${D14}/processed_co3d.h5"]=48000
        ["${D14}/processed_megadepth.h5"]=6000
        ["${D14}/processed_scannetpp.h5"]=6000
        ["${D14}/processed_BlendedMVS1_full.h5"]=6000
    )

    # Sort by samples/GB (computed once per file)
    local SORTED=()
    for src in "${!SOURCES[@]}"; do
        if [[ ! -e "$src" ]] && [[ ! -L "$src" ]]; then
            warn "Source not found: $src — skipping"
            continue
        fi
        # Skip already-staged (symlink pointing into NVME_DIR)
        if [[ -L "$src" ]]; then
            local target; target=$(readlink -f "$src" 2>/dev/null || true)
            if [[ "$target" == "${NVME_DIR}/"* ]]; then
                ok "Already staged: $(basename "$src")"
                continue
            fi
        fi
        SORTED+=("$src")
    done

    local SAFETY_MB=20480  # 20 GB safety margin

    for src in "${SORTED[@]}"; do
        local name; name=$(basename "$src")
        local dst="${NVME_DIR}/${name}"

        # Space check
        local free_mb; free_mb=$(df --output=avail -m "${NVME_DIR}" | tail -1 | tr -d ' ')
        local src_mb; src_mb=$(du -sm "$src" 2>/dev/null | cut -f1)
        local needed_mb=$(( src_mb + SAFETY_MB ))

        if (( free_mb < needed_mb )); then
            warn "Not enough NVMe space for ${name} (need ${needed_mb} MB, have ${free_mb} MB) — skipping"
            continue
        fi

        info "Staging ${name}  ($(( src_mb / 1024 )) GB)  →  ${NVME_DIR}"

        if [[ "${DRY_RUN}" == true ]]; then
            info "  DRY RUN — no files written"
            continue
        fi

        # Copy
        local t0; t0=$(date +%s)
        cp --no-preserve=timestamps "$src" "$dst"
        local t1; t1=$(date +%s)
        local elapsed=$(( t1 - t0 ))
        local speed=$(( src_mb / (elapsed > 0 ? elapsed : 1) ))
        ok "Copied in ${elapsed}s  (~${speed} MB/s)"

        # Integrity check (byte size)
        local src_bytes; src_bytes=$(stat --printf="%s" "$src")
        local dst_bytes; dst_bytes=$(stat --printf="%s" "$dst")
        if [[ "$src_bytes" != "$dst_bytes" ]]; then
            err "Size mismatch after copy: src=${src_bytes} dst=${dst_bytes} — removing dst"
            rm -f "$dst"
            continue
        fi

        # Symlink swap: rename HDD file → .hdd_original, create relative symlink
        local backup="${src}.hdd_original"
        mv "$src" "$backup"
        # Relative path from the source's directory to the NVMe destination
        # e.g.  disk14tb/X.h5 → ../datasetfaster/X.h5
        local rel_dst; rel_dst=$(python3 -c "
import os
print(os.path.relpath('${dst}', '$(dirname "${src}")'))
")
        ln -s "$rel_dst" "$src"
        ok "Staged: ${name}  (symlink: ${src} → ${rel_dst})"
    done

    # Summary
    echo ""
    local nvme_used; nvme_used=$(df --output=used -m "${NVME_DIR}" | tail -1 | tr -d ' ')
    local nvme_free; nvme_free=$(df --output=avail -m "${NVME_DIR}" | tail -1 | tr -d ' ')
    info "NVMe used: $(( nvme_used / 1024 )) GB  |  free: $(( nvme_free / 1024 )) GB"
}

# ── Dataset health check ──────────────────────────────────────────────────────
# VFS-aware: accepts either a directory or a basename-matching .h5 file.
check_path() {
    local p="$1"
    [[ -e "$p" ]] || [[ -e "${p}.h5" ]] || [[ -L "$p" ]]
}

check_datasets_stage2() {
    section "Dataset Health Check (Stage 2 — 31 datasets)"
    local missing=0

    declare -A DATASETS=(
        # ── disk14tb (corrected paths from audit) ─────────────────────────────
        ["Co3D"]="${D14}/processed_co3d"
        ["WildRGBD"]="${D14}/preprocess_wildrgbd"
        ["MegaDepth"]="${D14}/processed_megadepth"
        ["MapFree"]="${D14}/processed_mapfree"
        ["BlendedMVS1"]="${D14}/processed_BlendedMVS1_full"
        ["OmniObject3D"]="${D14}/processed_omniobject3d"
        ["ThreeDKenBurns"]="${D14}/processed_3dkb"
        ["SynScapes"]="${D14}/processed_synscapes"
        ["DynamicReplica"]="${D14}/processed_dynamic_replica"
        ["UASOL"]="${D14}/processed_uasol"
        ["MP3D"]="${D14}/matterport3d/processed_mp3d"
        # ── disk12tb ──────────────────────────────────────────────────────────
        ["ScanNetpp"]="${D12}/processed_scannetpp"
        ["DL3DV"]="${D12}/processed_dl3dv"
        ["WildRGBD_proc"]="${D12}/processed/preprocess_wildrgbd"
        ["UnReal4K"]="${D12}/processed/processed_unrealStereo4k"
        ["COP3D"]="${D12}/processed/processed_cop3d"
        ["MVImgNet"]="${D12}/processed/processed_mvimgnet"
        ["RE10K"]="${D12}/processed/processed_re10k"
        ["IRS"]="${D12}/processed_irs"
        ["EDEN"]="${D12}/processed_eden"
        ["Spring"]="${D12}/processed_spring"
        ["UrbanSyn"]="${D12}/processed_urbansyn"
        ["BEDLAM"]="${D12}/processed_bedlam"
        # ── diskPool ──────────────────────────────────────────────────────────
        ["ScanNet"]="${POOL}/processed_scannet"
        ["HyperSim"]="${POOL}/processed_hypersim"
        ["TartanAir"]="${POOL}/processed_TartanAir"
        ["VirtualKITTI2"]="${POOL}/vkitti_2.0.3_mixed"
        ["PointOdyssey"]="${POOL}/processed_point_odyssey"
        # ── fastdatasets (system NVMe) ─────────────────────────────────────────
        ["ARKitScenes"]="${FAST_DIR}/processed_arkitscenes"
        ["ARKitHighRes"]="${FAST_DIR}/processed_arkitscenes_highres"
        ["Waymo"]="${FAST_DIR}/waymo_processed"
        ["MVS_Synth"]="${FAST_DIR}/processed_mvs_synth"
    )

    for name in $(echo "${!DATASETS[@]}" | tr ' ' '\n' | sort); do
        local path="${DATASETS[$name]}"
        if check_path "$path"; then
            # Show NVMe tag if symlink points to datasetfaster
            if [[ -L "$path" ]] || [[ -L "${path}.h5" ]]; then
                local target; target=$(readlink -f "${path}" 2>/dev/null || readlink -f "${path}.h5" 2>/dev/null || "")
                if [[ "$target" == "${NVME_DIR}/"* ]]; then
                    ok "  [NVMe] ${name}"
                else
                    ok "  [HDD]  ${name}"
                fi
            else
                ok "  [HDD]  ${name}"
            fi
        else
            err "  MISSING  ${name}  →  ${path}"
            missing=$(( missing + 1 ))
        fi
    done

    # Pretrained checkpoint
    local ckpt="${CKPT_DIR}/recurrent_stage1/checkpoints/checkpoint-final.pth"
    if [[ -f "$ckpt" ]]; then
        ok "  Stage 1 checkpoint found"
    else
        err "  Stage 1 checkpoint MISSING: ${ckpt}"
        missing=$(( missing + 1 ))
    fi

    echo ""
    if (( missing > 0 )); then
        warn "${missing} path(s) missing — training may crash on those datasets."
    else
        ok "All 31 datasets + checkpoint verified."
    fi
    return $missing
}

# ── Stage runner ──────────────────────────────────────────────────────────────
run_stage() {
    local stage_n="$1"
    local config="recurrent_stage${stage_n}"
    local base_epochs extra_epochs target_epochs

    case "$stage_n" in
        1) base_epochs=6  ;;
        2) base_epochs=2  ;;
        3) base_epochs=2  ;;
        4) base_epochs=1  ;;
        *) err "Unknown stage: ${stage_n}"; exit 1 ;;
    esac

    extra_epochs="${RESUME_EXTRA}"
    target_epochs=$(( base_epochs + extra_epochs ))

    section "Stage ${stage_n} — config: ${config}  |  target epochs: ${target_epochs}"

    local overrides=("${PERF_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}")
    if (( extra_epochs > 0 )); then
        overrides+=("epochs=${target_epochs}")
        info "Resume mode: extending to ${target_epochs} epochs (auto_resume picks checkpoint-last.pth)"
    fi

    local cmd=("python" "${WORK_DIR}/src/train.py"
        "--config-name=${config}"
        "--config-path=${CONFIG_PATH}"
        "${overrides[@]+"${overrides[@]}"}")

    info "Command: ${cmd[*]}"
    info "Output:  ${CKPT_DIR}/${config}/"

    if [[ "${DRY_RUN}" == true ]]; then
        info "DRY RUN — training command printed above, not executed."
        return 0
    fi

    echo -e "${CYAN}──────────────────────────────────────────────────────────${NC}"
    local t0; t0=$(date +%s)

    set +e
    (cd "${WORK_DIR}" && "${cmd[@]}")
    local exit_code=$?
    set -e

    local t1; t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local h=$(( elapsed / 3600 ))
    local m=$(( (elapsed % 3600) / 60 ))
    local s=$(( elapsed % 60 ))

    echo -e "${CYAN}──────────────────────────────────────────────────────────${NC}"
    if [[ $exit_code -eq 0 ]]; then
        ok "Stage ${stage_n} finished in ${h}h ${m}m ${s}s"
    else
        err "Stage ${stage_n} exited with code ${exit_code}  (${h}h ${m}m ${s}s)"
        exit $exit_code
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}"
echo "  ██╗      ██████╗  ██████╗ ██████╗  ██████╗██╗   ██╗████████╗██████╗ "
echo "  ██║     ██╔═══██╗██╔═══██╗██╔══██╗██╔════╝██║   ██║╚══██╔══╝╚════██╗"
echo "  ██║     ██║   ██║██║   ██║██████╔╝██║     ██║   ██║   ██║    █████╔╝"
echo "  ██║     ██║   ██║██║   ██║██╔═══╝ ██║     ██║   ██║   ██║   ██╔═══╝ "
echo "  ███████╗╚██████╔╝╚██████╔╝██║     ╚██████╗╚██████╔╝   ██║   ███████╗"
echo "  ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝      ╚═════╝ ╚═════╝    ╚═╝   ╚══════╝"
echo -e "${NC}"
echo -e "  ${BOLD}Hardware:${NC}  RTX 5090  |  Ryzen 9 7800X3D  |  32 GB RAM"
echo -e "  ${BOLD}NVMe:${NC}      ${NVME_DIR}"
echo -e "  ${BOLD}Date:${NC}      $(date '+%Y-%m-%d %H:%M:%S')"

# GPU audit (always, even in dry-run)
gpu_audit

# NVMe staging
if [[ "${SKIP_STAGING}" == false ]]; then
    stage_to_nvme
else
    info "Skipping NVMe staging (--skip-staging)"
fi

# Dataset health check (stage 2 paths cover the superset)
check_datasets_stage2 || true   # warn but don't abort

# Determine which stages to run
if [[ "${ALL_STAGES}" == true ]]; then
    STAGES_TO_RUN=(2 3 4)
    info "Running stages: 2 → 3 → 4 (stage 1 assumed complete)"
else
    STAGES_TO_RUN=("${STAGE}")
fi

for s in "${STAGES_TO_RUN[@]}"; do
    run_stage "$s"
done

echo ""
ok "All done."
