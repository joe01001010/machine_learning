#!/usr/bin/env bash
# This bash script will spin up four role-specific LLM servers using vLLM
# Ensure python environment has vllm command
# Example: ./serve_models.sh Qwen/Qwen3-0.6B
# Example: ./serve-models.sh Qwen/Qwen3-4B-Instruct-2507

set -Eeuo pipefail
set +m


if (( $# != 1 )); then
    echo "ERROR: Must send the model you want all your agents to use"
    echo "Example: $0 Qwen/Qwen3-0.6B"
    echo "Example: $0 Qwen/Qwen3-4B-Instruct-2507"
    exit 67
fi


for executable in vllm setsid; do
    command -v "$executable" >/dev/null || {
        echo "ERROR: $executable is missing from the active environment." >&2
        exit 127
    }
done


base_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}
max_model_len=${MAX_MODEL_LEN:-8192}
# Qwen3-0.6B BF16 needs about 0.88 GiB of KV cache for 8192 tokens.
kv_cache_bytes=${KV_CACHE_MEMORY_BYTES:-1073741824}

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0


log_dir=$(mktemp -d .agent-logs.XXXXXX)
"$python_bin" "$base_dir/build_server_templates.py" "$1" "$1" "$1" "$1" --output "$log_dir/templates"
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
    echo "Starting server $role"
    setsid vllm serve "$model" \
        --chat-template "$log_dir/templates/$role.jinja" \
        --served-model-name "$role" \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --host 127.0.0.1 \
        --port "$port" \
        --dtype bfloat16 \
        --max-model-len "$max_model_len" \
        --max-num-seqs 1 \
        --gpu-memory-utilization "$budget" \
        --enforce-eager \
        --kv-cache-memory-bytes "$kv_cache_bytes" \
        --max-num-batched-tokens 512 \
        >"$log_dir/$role.log" 2>&1 &
    local pid=$!
    pids+=("$pid")
    roles["$pid"]=$role
    printf '%s: PID=%s port=%s log=%s\n' \
        "$role" "$pid" "$port" "$log_dir/$role.log"
}

start_server requirements-engineer "$1" 8001 0.25
start_server test-case-engineer    "$1" 8002 0.25
start_server test-engineer         "$1" 8003 0.25
start_server verification-engineer "$1" 8004 0.25

status=0
wait -n "${pids[@]}" || status=$?


echo "A model server exited (status $status); stopping the group."
for role in requirements-engineer test-case-engineer test-engineer verification-engineer; do
    echo "Last log lines for $role:"
    tail -n 12 "$log_dir/$role.log" || true
done
exit "$status"
