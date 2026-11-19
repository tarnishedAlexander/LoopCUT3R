#!/usr/bin/env bash
# Run CUT3R (reference) and LoopCUT3R over every example sequence and summarise
# the difference numerically.
#
# Each model has to be run from its own repo: the two trees have different
# src/dust3r packages and add_ckpt_path puts the local one on sys.path, so a
# single process cannot host both.
#
#   bash run_ab.sh                 # all sequences
#   bash run_ab.sh 002 004         # just these
#   FORCE=1 bash run_ab.sh         # re-extract even if the .npz already exists
#
# Env overrides: LOOPCK, CUTCK, OUT, EX, SIZE, and LOOP_MODEL_CFG (needed for
# stripped checkpoints such as checkpoint-final.pth, which carry weights only
# and no embedded args -- copy the model: line from that stage's YAML).
set -u

CUT3R_DIR=${CUT3R_DIR:-/app/CUT3R}
LOOP_DIR=${LOOP_DIR:-/app/recurrentTransformer/LoopCUT3R}
EX=${EX:-$LOOP_DIR/examples}
OUT=${OUT:-/app/recon}

CUTCK=${CUTCK:-$LOOP_DIR/checkpoints/cut3r_512_dpt_4_64.pth}
LOOPCK=${LOOPCK:-/app/recurrentTransformer/dataset/disk14tb/loopcut3r_ckpt/recurrent_stage4b_32v/checkpoints/checkpoint-stage4b-277k.pth}
SIZE=${SIZE:-512}

if [ $# -gt 0 ]; then SEQS=("$@")
else SEQS=(001 002 003 004_trunc16 004_trunc24 004_trunc32 004_trunc48 004); fi

mkdir -p "$OUT"
for f in "$CUTCK" "$LOOPCK"; do
  [ -f "$f" ] || { echo "missing checkpoint: $f" >&2; exit 1; }
done

extract () {  # repo  checkpoint  seq  outfile  [model_cfg]
  if [ -s "$4" ] && [ -z "${FORCE:-}" ]; then
    echo "    exists, skipping $(basename "$4")"; return 0
  fi
  local cfg=()
  [ -n "${5:-}" ] && cfg=(--model_cfg "$5")
  ( cd "$1" && python extract_recon.py \
      --model_path "$2" --seq_path "$EX/$3" --size "$SIZE" --out "$4" "${cfg[@]}" ) \
    > "$OUT/$(basename "${4%.npz}").log" 2>&1
  if [ -s "$4" ]; then
    echo "    ok  $(basename "$4")  ($(du -h "$4" | cut -f1))"
  else
    echo "    FAILED $(basename "$4") -- see $OUT/$(basename "${4%.npz}").log" >&2
    tail -5 "$OUT/$(basename "${4%.npz}").log" >&2
    return 1
  fi
}

for s in "${SEQS[@]}"; do
  if [ ! -d "$EX/$s" ]; then echo "skip $s (no such directory)"; continue; fi
  echo "== $s  ($(ls "$EX/$s" | wc -l) frames)"
  extract "$CUT3R_DIR" "$CUTCK" "$s" "$OUT/cut3r_$s.npz" || continue
  extract "$LOOP_DIR"  "$LOOPCK" "$s" "$OUT/loop_$s.npz" "${LOOP_MODEL_CFG:-}" || continue
done

echo
cd "$LOOP_DIR" && python ab_summary.py "$OUT" | tee "$OUT/summary.txt"
echo
echo "table saved to $OUT/summary.txt"
