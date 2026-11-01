"""Read the employee test dataset and validate distributed SQL results."""

import argparse
import json
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, IntegerType, LongType, StringType, StructField, StructType,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    types = {
        "long": LongType(), "integer": IntegerType(), "string": StringType(),
        "boolean": BooleanType(), "double": DoubleType(),
    }
    schema = StructType([StructField(item["name"], types[item["type"]], True) for item in manifest["schema"]])
    spark = SparkSession.builder.appName("Employees-1GiB").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    started = time.perf_counter()
    try:
        if not spark.sparkContext.master.startswith("spark://"):
            raise RuntimeError("Submit this test to the standalone Spark master, not local mode")
        df = spark.read.schema(schema).option("mode", "FAILFAST").json(args.data)
        print(f"Reading {manifest['total_bytes']:,} bytes across {df.rdd.getNumPartitions()} input partitions", flush=True)
        totals = df.agg(
            F.count("*").alias("rows"),
            F.sum("salary_cents").alias("sum_salary_cents"),
            F.sum("age").alias("sum_age"),
            F.sum(F.col("employed").cast("long")).alias("employed_count"),
        ).first().asDict()
        assert totals["rows"] == manifest["total_rows"], totals
        for key in ("sum_salary_cents", "sum_age", "employed_count"):
            assert totals[key] == manifest["expected"][key], (key, totals[key], manifest["expected"][key])
        print("Verified totals: " + json.dumps(totals, sort_keys=True), flush=True)

        # These groupings shuffle data across executors, while collecting only eight rows.
        grouped = df.groupBy("department").agg(
            F.count("*").alias("rows"),
            F.sum("salary_cents").alias("sum_salary_cents"),
            F.round(F.avg("age"), 2).alias("average_age"),
        )
        department_rows = grouped.orderBy("department").collect()
        assert len(department_rows) == len(manifest["by_department"]), department_rows
        print("Department aggregates:", flush=True)
        for row in department_rows:
            expected = manifest["by_department"][row["department"]]
            assert row["rows"] == expected["rows"] and row["sum_salary_cents"] == expected["sum_salary_cents"], row
            print(row.asDict(), flush=True)

        print("Average employed salary by country and department:", flush=True)
        df.filter("employed = true").groupBy("country", "department").agg(
            F.count("*").alias("employees"),
            F.round(F.avg("salary_cents") / 100, 2).alias("average_salary_usd"),
        ).orderBy(F.desc("average_salary_usd")).show(12, truncate=False)

        print("Highest-paid synthetic employees:", flush=True)
        df.select("employee_id", "name", "department", "country", "salary_cents").orderBy(
            F.desc("salary_cents"), "employee_id"
        ).show(10, truncate=False)
        elapsed = time.perf_counter() - started
        print(f"DATASET_TEST_SUCCESS: rows={totals['rows']} bytes={manifest['total_bytes']} elapsed_seconds={elapsed:.2f}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
