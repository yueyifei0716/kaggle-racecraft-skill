"""Check fixed-row CSV structure, ordered IDs and explicit numeric prediction columns."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def read_table(path, allow_empty=False):
    data = path.read_bytes()
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    header = next(reader, None)
    if not header or any(not name for name in header) or len(set(header)) != len(header):
        raise ValueError("missing or duplicate CSV column names")
    rows = list(reader)
    if not rows and not allow_empty or any(len(row) != len(header) or any(not cell.strip() for cell in row) for row in rows):
        raise ValueError("empty table, malformed row or blank cell")
    return header, rows, hashlib.sha256(data).hexdigest()


def check(sample, submission, ids, numbers, minimum=None, maximum=None, mode="fixed",
          label_columns=(), allowed_labels=(), probability_columns=()):
    expected, original, _ = read_table(sample)
    if mode not in ("fixed", "variable"):
        raise ValueError("mode must be fixed or variable")
    header, rows, digest = read_table(submission, allow_empty=mode == "variable")
    if header != expected or mode == "fixed" and len(rows) != len(original):
        raise ValueError("columns/order or row count differ from sample")
    if not ids or not (numbers or label_columns) or set(ids) & (set(numbers) | set(label_columns)):
        raise ValueError("specify disjoint ID and numeric prediction columns")
    if any(name not in header for name in list(ids) + list(numbers) + list(label_columns)):
        raise ValueError("requested column does not exist")
    if label_columns and not allowed_labels:
        raise ValueError("label columns require explicit allowed labels")
    if not set(probability_columns) <= set(numbers):
        raise ValueError("probability columns must also be numeric columns")
    if any(bound is not None and not math.isfinite(bound) for bound in (minimum, maximum)):
        raise ValueError("numeric bounds must be finite")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum exceeds maximum")
    id_indexes = [header.index(name) for name in ids]
    keys = [tuple(row[index] for index in id_indexes) for row in rows]
    sample_keys = [tuple(row[index] for index in id_indexes) for row in original]
    if len(set(keys)) != len(keys) or mode == "fixed" and keys != sample_keys:
        raise ValueError("IDs are duplicated or differ from sample order")
    for row in rows:
        for name in numbers:
            value = float(row[header.index(name)])
            if not math.isfinite(value):
                raise ValueError("non-finite prediction")
            if minimum is not None and value < minimum or maximum is not None and value > maximum:
                raise ValueError("prediction outside configured range")
        if any(row[header.index(name)] not in allowed_labels for name in label_columns):
            raise ValueError("prediction label outside allowed labels")
        if probability_columns:
            values = [float(row[header.index(name)]) for name in probability_columns]
            if any(not 0 <= value <= 1 for value in values) or not math.isclose(sum(values), 1, abs_tol=1e-6):
                raise ValueError("class probabilities must be in [0,1] and sum to one")
    return {"csv_check": "PASS", "rows": len(rows), "sha256": digest,
            "mode": mode, "scope": "CSV structure and configured values only; competition semantics require separate checks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--id-column", action="append", required=True)
    parser.add_argument("--numeric-column", action="append", default=[])
    parser.add_argument("--label-column", action="append", default=[])
    parser.add_argument("--allowed-label", action="append", default=[])
    parser.add_argument("--probability-column", action="append", default=[])
    parser.add_argument("--mode", choices=["fixed", "variable"], default="fixed")
    parser.add_argument("--minimum", type=float)
    parser.add_argument("--maximum", type=float)
    args = parser.parse_args()
    try:
        result = check(args.sample, args.submission, args.id_column, args.numeric_column, args.minimum, args.maximum,
                       args.mode, args.label_column, args.allowed_label, args.probability_column)
    except (ValueError, OSError, csv.Error, UnicodeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
