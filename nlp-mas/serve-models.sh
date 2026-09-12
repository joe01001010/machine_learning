#!/usr/bin/env bash
# This bash script will spin up three different LLMs using vLLM
# Ensure python environment has vllm command
# Example: ./serve_models.sh Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B
# Example: ./serve-models.sh Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507

set -Eeou pipefail
set +m


if (( $# != 3 )); then
    echo "Example: $0 ./serve_models.sh Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B"
    echo "Example: ./serve-models.sh Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507"
    exit 67
fi


command -v vllm >/dev/null
command -v setsid >/dev/null


export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0


log_dir=$(mktemp -d .agent-logs.XXXXXX)
declare -a pids=()
declare -A roles=()


cleanup() {
    local status=$?
    trap - EXIT
    trap '' INT TERM HUP

    echo
    echo "Stopping all model servers..."


    for pid in "${pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done


    local deadline=$((SECONDS + 20))
    while (( SECONDS < deadline )); do
        local alive=0
        for pid in "${pids[@]}"; do
            if kill -0 -- "-$pid" 2>/dev/null; then
                alive=1
            fi
        done
        if (( alive == 0 )); then
            break
        fi
        sleep 1
    done


    for pid in "${pids[@]}"; do
        if kill -0 -- "-$pid" 2>/dev/null; then
            echo "Force-stopping ${roles[$pid]}"
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    echo "Logs: $log_dir"
    exit "$status"
}


trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP


start_server() {
    local role=$1 model=$2 port=$3 budget=$4

    setsid vllm serve "$model" \
        --served-model-name "$role" \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --host 127.0.0.1 \
        --port "$port" \
        --dtype bfloat16 \
        --max-model-len 2048 \
        --max-num-seqs 1 \
        --gpu-memory-utilization "$budget" \
        --enforce-eager \
        --kv-cache-memory-bytes 536870912 \
        --max-num-batched-tokens 512 \
        >"$log_dir/$role.log" 2>&1 &

    local pid=$!
    pids+=("$pid")
    roles["$pid"]=$role

    printf '%s: PID=%s port=%s log=%s\n' \
        "$role" "$pid" "$port" "$log_dir/$role.log"
}


start_server requirements-engineer "$1" 8001 0.25
start_server test-engineer         "$2" 8002 0.35
start_server verification-engineer "$3" 8003 0.2


status=0
wait -n "${pids[@]}" || status=$?


echo "A model server exited (status $status); stopping the group."
exit "$status"
