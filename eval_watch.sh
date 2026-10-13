#!/bin/bash
# eval_watch.sh
# Watches r3 training log and launches eval_compare.sh automatically when
# the final test epoch completes.
#
# Usage:
#   bash eval_watch.sh &          # run in background, check with: tail -f eval_watch.log
#   bash eval_watch.sh --now      # skip waiting, eval current checkpoint-last immediately
#
# The script checks every 60 seconds. Kill it with: kill $(cat eval_watch.pid)

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${WORKDIR}/checkpoints/recurrent_stage1_r3/train.log"
WATCH_LOG="${WORKDIR}/eval_watch.log"
PID_FILE="${WORKDIR}/eval_watch.pid"
TOTAL_EPOCHS=15    # must match config epochs: 15  (last test epoch = [15])
FINAL_SIGNAL="Test Epoch: \[${TOTAL_EPOCHS}\] Total time"

cd "$WORKDIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WATCH_LOG"; }

# ── immediate mode ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--now" ]]; then
    log "Running eval immediately on current checkpoint-last.pth"
    bash eval_compare.sh 2>&1 | tee -a "$WATCH_LOG"
    exit 0
fi

# ── background watch mode ─────────────────────────────────────────────────────
echo $$ > "$PID_FILE"

log "eval_watch started (PID=$$)"
log "Watching: $LOG_FILE"
log "Trigger : '$FINAL_SIGNAL'"
log "Kill with: kill \$(cat eval_watch.pid)"
log ""

while true; do
    if [[ ! -f "$LOG_FILE" ]]; then
        log "Log file not found yet, waiting..."
        sleep 60
        continue
    fi

    if grep -qP "$FINAL_SIGNAL" "$LOG_FILE" 2>/dev/null; then
        log "━━━ Training complete! Test Epoch [${TOTAL_EPOCHS}] detected ━━━"
        log "Waiting 30s for checkpoint to be written..."
        sleep 30

        CKPT="${WORKDIR}/checkpoints/recurrent_stage1_r3/checkpoints/checkpoint-last.pth"
        if [[ ! -f "$CKPT" ]]; then
            log "ERROR: checkpoint-last.pth not found at $CKPT"
            exit 1
        fi

        log "Starting benchmark comparison..."
        bash "${WORKDIR}/eval_compare.sh" 2>&1 | tee -a "$WATCH_LOG"
        log "Done. Results in eval_results/"
        rm -f "$PID_FILE"
        exit 0
    fi

    # Show training progress while waiting
    LAST_LINE="$(grep "Epoch: \[" "$LOG_FILE" | tail -1 | grep -oP 'Epoch: \[\d+\]\s+\[\d+/\d+\].*?eta: [^ ]+' || true)"
    if [[ -n "$LAST_LINE" ]]; then
        log "Training: $LAST_LINE"
    fi

    sleep 60
done
