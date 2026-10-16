# Apache Spark on Minikube: one master and four workers

This chart creates an `apache-spark` namespace and deploys a Spark standalone cluster inside it. It is separate from your earlier single-job chart.

Verified on October 15, 2026: the namespace was created by Helm, the master was ready, all four workers were registered ALIVE, and a separate PySpark driver successfully ran your employee example plus Python tasks across all four workers. All five cluster pods had zero restarts. The temporary validation Job was removed after saving its logs.

```text
Minikube profile / Kubernetes context: pyspark
  default namespace
    Helm release record: apache-spark
  apache-spark namespace (created by this chart)
    master Deployment: 1 pod
    worker Deployment: 4 pods
    master Service: ports 7077 and 8080
```

All five Spark pods run on the existing Minikube node. Four worker pods provide separate Spark worker processes; they do not create four Kubernetes nodes.

## 1. Namespace creation belongs to the chart

The default settings are:

```yaml
namespace:
  create: true
  name: apache-spark

worker:
  replicas: 4
```

`templates/namespace.yaml` creates the Namespace resource. Every Service and Deployment explicitly sets `metadata.namespace` from `namespace.name`.

The Helm release record lives in the existing `default` namespace. Helm stores that record before installing chart resources, so the release's own namespace must already exist. This arrangement lets the chart itself create the Spark workload namespace without requiring `--create-namespace` on the command line.

Use `--namespace default` with Helm, and `-n apache-spark` with kubectl when inspecting Spark.

## 2. The resources and their roles

| Template | Purpose |
| --- | --- |
| `templates/namespace.yaml` | Creates `apache-spark`; preserves it on uninstall. |
| `templates/master-deployment.yaml` | Keeps one master daemon running. |
| `templates/master-service.yaml` | Gives workers and clients a stable DNS name for the master. |
| `templates/worker-deployment.yaml` | Keeps four worker daemons running through `replicas: 4`. |

The Spark master allocates resources to applications. Workers launch executors that run application tasks. Each application's driver coordinates its work. This chart supplies the master and workers; application drivers are submitted separately.

Deployments fit these long-running daemons. A Service is useful for the master because a replacement master pod can receive a new IP. Workers connect to the stable name `apache-spark-master:7077` within the namespace. Their advertised addresses use their individual pod IPs.

The master's `Recreate` update strategy keeps it to one master during replacement. This learning setup has no replicated master state, so master restarts or upgrades can interrupt active applications.

The master Service is internal (`ClusterIP`). The Spark UI can be reached through port forwarding when you want to inspect it.

## 3. Install or update

The initial installation has already been applied. These Ubuntu / WSL commands reproduce it or apply later edits:

```bash
cd /home/weibelj/git/machine_learning/pyspark

helm lint ./charts/apache-spark --strict

helm template apache-spark ./charts/apache-spark --namespace default

helm upgrade --install apache-spark ./charts/apache-spark \
  --kube-context pyspark \
  --namespace default \
  --wait --timeout 5m
```

`helm template` previews the Namespace, Service, and two Deployments without applying them. `upgrade --install` installs the release if absent or updates it when present. The resource labels and Service selector connect the master Service to the master pod.

If using a namespace that someone already manages, set `namespace.create=false`. Do not change the namespace of an existing release as a routine update; use a separate release for a different namespace.

## 4. Inspect the cluster

There is no standalone kubectl on your Ubuntu PATH, so these commands use Minikube's copy:

```bash
minikube -p pyspark kubectl -- --context=pyspark get namespace apache-spark

minikube -p pyspark kubectl -- --context=pyspark -n apache-spark \
  get deployments,pods,services
```

Expected steady state:

```text
apache-spark-master   1/1 ready
apache-spark-worker   4/4 ready
```

Inspect worker registration in the master logs or master UI:

```bash
minikube -p pyspark kubectl -- --context=pyspark -n apache-spark \
  logs deployment/apache-spark-master

minikube -p pyspark kubectl -- --context=pyspark -n apache-spark \
  port-forward service/apache-spark-master 8080:8080
```

Keep the forwarding terminal open and visit [the master UI](http://localhost:8080). It should list four ALIVE workers.

An application driver running elsewhere inside the cluster can use:

```text
spark://apache-spark-master.apache-spark.svc.cluster.local:7077
```

## 5. Why these resource settings?

Each worker advertises **1 Spark core and 512m of executor memory**. Four workers provide four Spark cores and 2 GiB of advertised executor memory in total. These are small learning settings.

Spark executor memory differs from the container's memory limit. Each worker container has a 1Gi limit to include the executor JVM, worker daemon, Python processes, and other overhead. Its worker daemon has a separate 128m heap. The master daemon uses a 256m heap and a 512Mi container limit. Larger jobs may require larger limits and more Minikube memory.

CPU requests help Kubernetes schedule the pods; CPU limits cap their usage. Four workers on one Minikube node share that node's hardware.

`SPARK_NO_DAEMONIZE=true` keeps each Spark process in the foreground so Kubernetes can monitor it. Probes check that each daemon is responding; verifying all four workers in the master's registry checks that they actually connected. Neither daemon needs a Kubernetes API token in this standalone layout.

## 6. Lifecycle and the next step

The namespace has the `helm.sh/resource-policy: keep` annotation. Running:

```bash
helm uninstall apache-spark --kube-context pyspark --namespace default
```

removes the Deployments and Service but leaves the namespace. This avoids deleting the namespace and any separately submitted application resources in it. Reinstalling the same release in the same release namespace can reuse the retained namespace.

The next step is submitting your employee script from a separate driver pod, using `spark-submit --master spark://apache-spark-master:7077 --deploy-mode client`. For PySpark with a standalone master, use client deploy mode. The driver pod needs an address reachable by worker executors; a laptop's localhost address will not work for that connection.

## Official references

- [Spark standalone master, workers, and resource settings](https://spark.apache.org/docs/4.2.0/spark-standalone.html)
- [Spark application submission and Python deploy modes](https://spark.apache.org/docs/4.2.0/submitting-applications.html)
- [Kubernetes namespaces](https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/)
- [Helm resource retention annotation](https://helm.sh/docs/howto/charts_tips_and_tricks/)
