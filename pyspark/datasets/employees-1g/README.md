# One shared employee data file

`employees.jsonl` is exactly **1 GiB**, with **3,360,001 synthetic employees**. The original 32 files in `data/` contain the same records. The single file has already been uploaded to your Minikube node.

The Helm chart mounts `/data/apache-spark` from the Minikube node as `/data` in every worker. There is one physical cluster copy, shared by the workers.

To upload a replacement file from Ubuntu:

```bash
minikube -p minikube cp employees.jsonl /data/apache-spark/employees.jsonl
```

PySpark code running inside a worker can read it with:

```python
df = spark.read.json("file:///data/employees.jsonl")
df.count()
```

To run the included aggregation test:

```bash
cd /home/weibelj/git/machine_learning/pyspark/datasets/employees-1g
bash ./run_spark_test.sh
```

The updated runner copies the single file once and submits the benchmark from an existing worker. It uses no temporary driver pod and no copies inside individual workers. The benchmark checks counts, totals, and department aggregates against `manifest.json`.

This shared node folder fits your single-node Minikube setup. Deleting the Minikube profile deletes its copy; the source file remains in your project. Spark's [file access documentation](https://spark.apache.org/docs/4.2.0/rdd-programming-guide.html#external-datasets) explains why every executor needs access to the data.
