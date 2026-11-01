#!/usr/bin/env bash
# This bash script will spin up four role-specific LLM servers using vLLM
# Ensure python environment has vllm command
# Example: ./serve-models.sh Eslzzyl/Qwen3-4B-Instruct-2507-AWQ
# Example: ./serve-models.sh Qwen/Qwen3-4B-Instruct-2507

set -Eeuo pipefail
set +m


if (( $# != 1 )); then
    echo "ERROR: Must send the model you want all your agents to use"
    echo "4-bit Example: $0 Eslzzyl/Qwen3-4B-Instruct-2507-AWQ"
    echo "16-bit Example: $0 Qwen/Qwen3-4B-Instruct-2507"
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
# Four 4B copies require 4-bit weights on a 16 GiB GPU.
# Qwen3-4B needs about 297 MiB of dynamically scaled FP8 KV for 4096 tokens.
max_model_len=${MAX_MODEL_LEN:-4096}
kv_cache_bytes=${KV_CACHE_MEMORY_BYTES:-335544320}
kv_cache_dtype=${KV_CACHE_DTYPE:-fp8_per_token_head}
attention_backend=${ATTENTION_BACKEND:-TRITON_ATTN}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.20}
startup_timeout=${STARTUP_TIMEOUT:-600}
server_roles=(requirements-engineer test-engineer coder verification-engineer)

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0


# Reject occupied ports before a different server's /health can imply readiness.
"$python_bin" - <<'PYCHECK'
from contextlib import ExitStack
import socket

with ExitStack() as stack:
    for port in range(8001, 8005):
        listener = stack.enter_context(socket.socket())
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"ERROR: Cannot use port {port}: {exc}. Stop any existing server first.")
PYCHECK

log_dir=$(mktemp -d "$base_dir/.agent-logs.XXXXXX")
"$python_bin" "$base_dir/build_server_templates.py" "$1" "$1" "$1" "$1" \
    --roles "${server_roles[@]}" --output "$log_dir/templates"
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
        --kv-cache-dtype "$kv_cache_dtype" \
        --attention-backend "$attention_backend" \
        --max-num-batched-tokens 512 \
        >"$log_dir/$role.log" 2>&1 &
    local pid=$!
    pids+=("$pid")
    roles["$pid"]=$role
    printf '%s: PID=%s port=%s log=%s\n' \
        "$role" "$pid" "$port" "$log_dir/$role.log"
}

wait_for_server() {
    local role=$1 port=$2 deadline=$((SECONDS + startup_timeout))
    echo "Waiting for $role on port $port..."
    while ! "$python_bin" -c '
import sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:" + sys.argv[1] + "/health", timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
' "$port"; do
        for pid in "${pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "ERROR: ${roles[$pid]} exited during startup." >&2
                tail -n 40 "$log_dir/${roles[$pid]}.log" >&2
                return 1
            fi
        done
        if (( SECONDS >= deadline )); then
            echo "ERROR: $role did not become ready within $startup_timeout seconds." >&2
            tail -n 40 "$log_dir/$role.log" >&2
            return 1
        fi
        sleep 2
    done
    echo "$role is ready."
}

# Initialize one at a time so GPU allocations do not race during profiling.
for i in "${!server_roles[@]}"; do
    port=$((8001 + i))
    start_server "${server_roles[$i]}" "$1" "$port" "$gpu_memory_utilization"
    wait_for_server "${server_roles[$i]}" "$port"
done
echo "All four model servers are ready. Logs: $log_dir"

status=0
wait -n "${pids[@]}" || status=$?


echo "A model server exited (status $status); stopping the group."
for role in "${server_roles[@]}"; do
    echo "Last log lines for $role:"
    tail -n 12 "$log_dir/$role.log" || true
done
exit "$status"
