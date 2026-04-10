#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-docker-dev-1}"
WORKDIR="${WORKDIR:-/app/recurrentTransformer/LoopCUT3R}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-120}"
TEST_SAMPLES="${TEST_SAMPLES:-20}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_ITER="${ACCUM_ITER:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PRINT_FREQ="${PRINT_FREQ:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints_smoke}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/loop_ab_smoke}"

ORIGINAL_LOG="${LOG_PREFIX}_original.log"
LOOP_NEW_LOG="${LOG_PREFIX}_loop_new.log"

declare -A WALL_SECS

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker command not found." >&2
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "ERROR: container '${CONTAINER_NAME}' is not running." >&2
  exit 1
fi

run_variant() {
  local config_name="$1"
  local variant="$2"
  local tag="$3"
  local log_file="$4"

  local start end
  start=$(date +%s)

  docker exec -w "${WORKDIR}" "${CONTAINER_NAME}" \
    python src/train.py \
      --config-name "${config_name}" \
      --variant "${variant}" \
      epochs="${EPOCHS}" \
      eval_freq=0 \
      "train_dataset=${TRAIN_SAMPLES} @ \${dataset1}" \
      "test_dataset=${TEST_SAMPLES} @ \${dataset1}" \
      batch_size="${BATCH_SIZE}" \
      accum_iter="${ACCUM_ITER}" \
      num_workers="${NUM_WORKERS}" \
      print_freq="${PRINT_FREQ}" \
      save_freq=1000 \
      keep_freq=1000 \
      output_dir="${OUTPUT_ROOT}/${tag}" \
      > "${log_file}" 2>&1

  end=$(date +%s)
  WALL_SECS["${tag}"]=$((end - start))
  echo "${tag}: wall_seconds=${WALL_SECS[${tag}]}"
}

run_variant "recurrent_dual_variant_original_cut3r" "original_cut3r" "original" "${ORIGINAL_LOG}"
run_variant "recurrent_dual_variant_loop_new" "loop_new" "loop_new" "${LOOP_NEW_LOG}"

export ORIGINAL_LOG
export LOOP_NEW_LOG
export ORIGINAL_WALL_SECS="${WALL_SECS[original]}"
export LOOP_NEW_WALL_SECS="${WALL_SECS[loop_new]}"

python - <<'PY'
import os
import re
from pathlib import Path

paths = {
    "original_cut3r": Path(os.environ["ORIGINAL_LOG"]),
    "loop_new": Path(os.environ["LOOP_NEW_LOG"]),
}
wall = {
    "original_cut3r": int(os.environ["ORIGINAL_WALL_SECS"]),
    "loop_new": int(os.environ["LOOP_NEW_WALL_SECS"]),
}

pat_avg = re.compile(r"Averaged stats: .*?loss:\s*([0-9.]+)\s*\(([0-9.]+)\)")
pat_epoch_time = re.compile(r"Epoch: \[(\d+)\] Total time: .*?\(([0-9.]+) s / it\)")
pat_step_end = re.compile(r"Epoch: \[(\d+)\].*?\[59/60\].*?loss:\s*([0-9.]+)\s*\(([0-9.]+)\).*?max mem:\s*([0-9]+)")
pat_train_time = re.compile(r"Training time\s+([0-9:]+)")

summary = {}
for name, path in paths.items():
    text = path.read_text(errors="ignore")

    avg_lines = pat_avg.findall(text)
    if not avg_lines:
        raise RuntimeError(f"No 'Averaged stats' line found in {path}")
    final_last_loss = float(avg_lines[-1][0])
    final_avg_loss = float(avg_lines[-1][1])

    epoch_times = [float(s) for _, s in pat_epoch_time.findall(text)]
    if not epoch_times:
        raise RuntimeError(f"No per-epoch iteration timing lines found in {path}")
    mean_s_per_it = sum(epoch_times) / len(epoch_times)

    step_lines = [(int(e), float(last), float(avg), int(mem)) for e, last, avg, mem in pat_step_end.findall(text)]
    if not step_lines:
        raise RuntimeError(f"No [59/60] lines found in {path}")
    step_lines.sort(key=lambda x: x[0])
    _, final_step_last, final_step_avg, final_mem = step_lines[-1]

    tt = pat_train_time.findall(text)
    train_time = tt[-1] if tt else "n/a"

    summary[name] = {
        "mean_s_per_it": mean_s_per_it,
        "final_avg_loss": final_avg_loss,
        "final_last_loss": final_last_loss,
        "final_step_last": final_step_last,
        "final_step_avg": final_step_avg,
        "final_max_mem": final_mem,
        "training_time": train_time,
        "wall_seconds": wall[name],
    }

print("\nVariant\twall_s\ttrain_time\tsec_per_it\tfinal_avg_loss\tfinal_last_loss\tfinal_max_mem")
for name in ("original_cut3r", "loop_new"):
    s = summary[name]
    print(
        f"{name}\t{s['wall_seconds']}\t{s['training_time']}\t"
        f"{s['mean_s_per_it']:.4f}\t{s['final_avg_loss']:.4f}\t"
        f"{s['final_last_loss']:.4f}\t{s['final_max_mem']}"
    )

orig = summary["original_cut3r"]
loop = summary["loop_new"]
print("\nDeltas (loop_new vs original_cut3r)")
print(f"speed_pct={(loop['mean_s_per_it'] / orig['mean_s_per_it'] - 1) * 100:.2f}")
print(f"mem_mb={loop['final_max_mem'] - orig['final_max_mem']}")
print(f"final_avg_loss={loop['final_avg_loss'] - orig['final_avg_loss']:.4f}")
print(f"final_last_loss={loop['final_last_loss'] - orig['final_last_loss']:.4f}")
PY

printf "\nLogs saved to:\n  %s\n  %s\n" "${ORIGINAL_LOG}" "${LOOP_NEW_LOG}"
