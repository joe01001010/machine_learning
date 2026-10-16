"""Independently check file sizes, hashes, JSON records, IDs, and aggregate totals."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def verify_part(task):
    root, item, schema, departments = task
    path = Path(root) / item["path"]
    assert path.stat().st_size == item["bytes"], path
    digest = hashlib.sha256()
    rows = salary = ages = employed = 0
    grouped = {department: {"rows": 0, "sum_salary_cents": 0} for department in departments}
    fields = {field["name"] for field in schema}
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            assert line.endswith(b"\n"), path
            record = json.loads(line)
            assert set(record) == fields, (path, rows)
            assert record["employee_id"] == item["first_employee_id"] + rows, (path, rows)
            assert 18 <= record["age"] <= 70 and type(record["employed"]) is bool, (path, rows)
            assert record["salary_cents"] >= 0, (path, rows)
            assert record["employed"] or record["salary_cents"] == 0, (path, rows)
            rows += 1
            salary += record["salary_cents"]
            ages += record["age"]
            employed += int(record["employed"])
            grouped[record["department"]]["rows"] += 1
            grouped[record["department"]]["sum_salary_cents"] += record["salary_cents"]
    assert rows == item["rows"], path
    assert digest.hexdigest() == item["sha256"], path
    assert item["last_employee_id"] == item["first_employee_id"] + rows - 1, path
    return {"rows": rows, "sum_salary_cents": salary, "sum_age": ages, "employed_count": employed, "grouped": grouped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert {path.name for path in (root / "data").iterdir()} == {Path(item["path"]).name for item in manifest["files"]}
    assert sum(item["bytes"] for item in manifest["files"]) == manifest["total_bytes"]
    tasks = [(str(root), item, manifest["schema"], list(manifest["by_department"])) for item in manifest["files"]]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(verify_part, tasks))
    assert sum(result["rows"] for result in results) == manifest["total_rows"]
    for key in ("sum_salary_cents", "sum_age", "employed_count"):
        assert sum(result[key] for result in results) == manifest["expected"][key], key
    for department, expected in manifest["by_department"].items():
        for key in ("rows", "sum_salary_cents"):
            assert sum(result["grouped"][department][key] for result in results) == expected[key], (department, key)
    print(json.dumps({"status": "verified", "rows": manifest["total_rows"], "bytes": manifest["total_bytes"], "files": len(results), "checks": ["all JSON records parsed", "unique sequential IDs within non-overlapping file ranges", "SHA256 per file", "exact file sizes", "row counts", "age totals", "employment totals", "salary totals", "department aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
