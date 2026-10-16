"""Generate deterministic, partitioned JSONL employee data using only Python's stdlib."""

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

DEPARTMENTS = (
    "Engineering", "Sales", "Marketing", "Finance", "Operations", "Support",
    "Human Resources", "Research",
)
COUNTRIES = ("US", "CA", "GB", "DE", "AU", "IN")
NOTES = (
    "Synthetic record for distributed analytics and Spark cluster testing.",
    "Completed annual training and participated in a cross-functional project.",
    "Assigned to a regional team with a quarterly performance review schedule.",
    "Sample employee profile generated for filtering, sorting, and aggregation.",
    "Participates in mentoring, training, and internal process improvements.",
    "Works with customers and colleagues across several operational teams.",
)
SCHEMA = [
    {"name": "employee_id", "type": "long"},
    {"name": "name", "type": "string"},
    {"name": "department", "type": "string"},
    {"name": "country", "type": "string"},
    {"name": "office_id", "type": "string"},
    {"name": "age", "type": "integer"},
    {"name": "employed", "type": "boolean"},
    {"name": "salary_cents", "type": "long"},
    {"name": "tenure_years", "type": "integer"},
    {"name": "performance_score", "type": "double"},
    {"name": "last_review_date", "type": "string"},
    {"name": "notes", "type": "string"},
]


def encode(record):
    return (json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def generate_part(task):
    root, index, target_bytes, seed = task
    path = Path(root) / "data" / f"part-{index:05d}.jsonl"
    rng = random.Random(seed + index)
    digest = hashlib.sha256()
    rows = written = salary_total = age_total = employed_count = 0
    by_department = {name: {"rows": 0, "sum_salary_cents": 0} for name in DEPARTMENTS}
    pending = bytearray()

    def flush(handle):
        if pending:
            handle.write(pending)
            digest.update(pending)
            pending.clear()

    with path.open("xb", buffering=4 * 1024 * 1024) as handle:
        while written < target_bytes:
            employee_id = index * 10_000_000 + rows
            department = rng.choice(DEPARTMENTS)
            country = rng.choice(COUNTRIES)
            age = rng.randint(18, 70)
            employed = rng.random() < 0.88
            salary = rng.randint(3_500_000, 22_000_000) if employed else 0
            record = {
                "employee_id": employee_id,
                "name": f"Employee_{employee_id:010d}",
                "department": department,
                "country": country,
                "office_id": f"{country}-{rng.randrange(100):03d}",
                "age": age,
                "employed": employed,
                "salary_cents": salary,
                "tenure_years": rng.randint(0, min(age - 18, 40)),
                "performance_score": rng.randint(10, 50) / 10,
                "last_review_date": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                "notes": rng.choice(NOTES),
            }
            remaining = target_bytes - written
            if remaining <= 1024:
                # Make the LAST complete JSON record fill the file exactly.
                # All other records contain ordinary short synthetic notes.
                record["notes"] = ""
                base_size = len(encode(record))
                padding = remaining - base_size
                if padding < 0:
                    raise ValueError(f"Insufficient room for the last record: {remaining}")
                record["notes"] = ("synthetic " * ((padding + 9) // 10))[:padding]
                line = encode(record)
                assert len(line) == remaining
            else:
                line = encode(record)
                assert len(line) < 512

            pending.extend(line)
            written += len(line)
            rows += 1
            salary_total += salary
            age_total += age
            employed_count += int(employed)
            by_department[department]["rows"] += 1
            by_department[department]["sum_salary_cents"] += salary
            if len(pending) >= 1024 * 1024:
                flush(handle)
        flush(handle)

    assert path.stat().st_size == target_bytes
    return {
        "path": f"data/{path.name}", "bytes": written, "rows": rows,
        "sha256": digest.hexdigest(), "sum_salary_cents": salary_total,
        "sum_age": age_total, "employed_count": employed_count,
        "by_department": by_department,
        "first_employee_id": index * 10_000_000,
        "last_employee_id": index * 10_000_000 + rows - 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=1024**3)
    parser.add_argument("--parts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.parts < 1 or args.bytes < args.parts * 2048 or args.workers < 1:
        parser.error("Choose positive workers/parts and at least 2048 bytes per part")
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    data_dir.mkdir(exist_ok=True)
    if any(data_dir.iterdir()) or (args.output / "manifest.json").exists():
        parser.error("Output already contains data or a manifest; choose a new output directory")
    started = perf_counter()
    quotient, remainder = divmod(args.bytes, args.parts)
    tasks = [(str(args.output), i, quotient + (i < remainder), args.seed) for i in range(args.parts)]
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, args.parts)) as pool:
        futures = [pool.submit(generate_part, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Generated {len(results):02d}/{args.parts}: {result['path']} ({result['rows']:,} rows)", flush=True)
    results.sort(key=lambda result: result["path"])
    by_department = {
        dept: {
            key: sum(result["by_department"][dept][key] for result in results)
            for key in ("rows", "sum_salary_cents")
        }
        for dept in DEPARTMENTS
    }
    manifest = {
        "dataset": "synthetic-employees-1g", "format": "jsonl", "encoding": "UTF-8",
        "compression": "none", "seed": args.seed,
        "size_definition": "1 GiB = 1,073,741,824 bytes of data files, excluding metadata",
        "total_bytes": sum(result["bytes"] for result in results),
        "total_rows": sum(result["rows"] for result in results),
        "part_count": args.parts, "schema": SCHEMA,
        "expected": {
            key: sum(result[key] for result in results)
            for key in ("sum_salary_cents", "sum_age", "employed_count")
        },
        "by_department": by_department,
        "notes": [
            "Entirely synthetic data; names are generated identifiers.",
            "salary_cents is an annual salary expressed as integer USD cents; unemployed rows have zero salary.",
            "Employee IDs are unique with separate ranges for each file; they are not contiguous across files.",
            "The notes field in each file's final record is adjusted to make the file exactly the target size.",
        ],
        "files": [
            {key: result[key] for key in ("path", "bytes", "rows", "sha256", "first_employee_id", "last_employee_id")}
            for result in results
        ],
    }
    assert manifest["total_bytes"] == args.bytes
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {manifest['total_rows']:,} rows; {manifest['total_bytes']:,} bytes; {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
