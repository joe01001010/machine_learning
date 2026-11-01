#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONTEXT=${KUBE_CONTEXT:-minikube}
NAMESPACE=${NAMESPACE:-apache-spark}
RELEASE=${RELEASE_NAME:-apache-spark}

if command -v kubectl >/dev/null 2>&1; then
  KUBE=(kubectl --context "$CONTEXT" -n "$NAMESPACE")
else
  KUBE=(minikube -p "$CONTEXT" kubectl -- --context "$CONTEXT" -n "$NAMESPACE")
fi

# One data copy to the shared directory on the Minikube node.
minikube -p "$CONTEXT" cp "$HERE/employees.jsonl" /data/apache-spark/employees.jsonl
minikube -p "$CONTEXT" cp "$HERE/manifest.json" /data/apache-spark/manifest.json
minikube -p "$CONTEXT" cp "$HERE/benchmark_employees.py" /data/apache-spark/benchmark_employees.py

# Run the Python driver in an existing worker, which already mounts /data.
"${KUBE[@]}" exec "deployment/$RELEASE-worker" -- sh -c '
  exec /opt/spark/bin/spark-submit \
    --master "$1" --deploy-mode client \
    --driver-memory 512m --executor-memory 512m \
    --executor-cores 1 --total-executor-cores 4 \
    --conf "spark.driver.host=$POD_IP" \
    --conf spark.driver.bindAddress=0.0.0.0 \
    --conf spark.scheduler.minRegisteredResourcesRatio=1.0 \
    --conf spark.sql.shuffle.partitions=32 \
    /data/benchmark_employees.py \
    --data file:///data/employees.jsonl --manifest /data/manifest.json
' sh "spark://$RELEASE-master:7077"
