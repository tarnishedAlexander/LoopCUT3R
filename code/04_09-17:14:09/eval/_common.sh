#!/bin/bash

cut3r_resolve_eval_weights() {
    local workdir="${1:-.}"
    local default_weights="${2:-${workdir}/src/cut3r_512_dpt_4_64.pth}"
    local fallback_weights="${3:-${workdir}/src/checkpoints/decoder_cka_finetune_v1/checkpoints/checkpoint-last.pth}"
    local model_weights="${CUT3R_EVAL_WEIGHTS:-$default_weights}"

    if [[ ! -f "$model_weights" ]]; then
        model_weights="$fallback_weights"
    fi

    if [[ ! -f "$model_weights" ]]; then
        echo "Error: no usable checkpoint found." >&2
        echo "Tried: ${CUT3R_EVAL_WEIGHTS:-$default_weights}" >&2
        echo "Fallback: $fallback_weights" >&2
        return 1
    fi

    printf '%s\n' "$model_weights"
}


cut3r_print_eval_checkpoint() {
    local model_weights="$1"
    echo "Using checkpoint: $model_weights"
}


cut3r_accelerate_launch() {
    local default_num_processes="$1"
    local default_main_process_port="$2"
    shift 2

    local num_processes="${CUT3R_EVAL_NUM_PROCESSES:-$default_num_processes}"
    local main_process_port="${CUT3R_EVAL_MAIN_PROCESS_PORT:-$default_main_process_port}"

    accelerate launch \
        --num_processes "$num_processes" \
        --main_process_port "$main_process_port" \
        "$@"
}
