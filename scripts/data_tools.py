"""Persistent, deterministic competition splits without third-party dependencies."""

import argparse
import csv
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def read_rows(path):
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"), newline=""))
    names = reader.fieldnames
    if not names or len(set(names)) != len(names):
        raise ValueError("missing or repeated column names")
    rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("empty or malformed input")
    return names, rows


def write_new(path, names, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def assign(rows, id_column, strategy, splits, seed=17, group=None, target=None,
           time_column=None, gap=0, regression_bins=None):
    if strategy == "leave-group-out":
        if not group:
            raise ValueError("leave-group-out requires --group")
        splits = len({row[group] for row in rows})
    if splits < 2 or gap < 0:
        raise ValueError("need >=2 folds and non-negative time gap")
    if any(id_column not in row or not row[id_column] for row in rows):
        raise ValueError("missing sample IDs")
    ids = [row[id_column] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique")
    if any("fold" in row for row in rows):
        raise ValueError("fold already exists; do not overwrite a comparison surface")
    if strategy in ("group", "stratified-group", "leave-group-out") and not group:
        raise ValueError("group strategy requires --group")
    if strategy in ("stratified", "stratified-group") and not target:
        raise ValueError("stratified strategy requires --target")
    if strategy == "time":
        if not time_column or group or regression_bins:
            raise ValueError("time mode requires --time and disallows group/bins")
        buckets = defaultdict(list)
        time_modes = set()
        for index, row in enumerate(rows):
            value = row[time_column]
            if not value:
                raise ValueError("empty time value")
            try:
                point = float(value)
                time_modes.add("numeric")
            except ValueError:
                time_modes.add("iso")
                moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if moment.tzinfo is None:
                    raise ValueError("text timestamps require an explicit timezone")
                point = moment.timestamp()
            if not math.isfinite(point):
                raise ValueError("non-finite time value")
            buckets[point].append(index)
        if len(time_modes) != 1:
            raise ValueError("time column mixes numeric and ISO values; normalize it before splitting")
        ordered = sorted(buckets)
        if len(ordered) < splits + 1:
            raise ValueError("time split needs training history plus validation blocks")
        fold = [-1] * len(rows)
        blocks = [[] for _ in range(splits + 1)]
        for position, value in enumerate(ordered):
            block = min(splits, position * (splits + 1) // len(ordered))
            blocks[block].append(value)
            for index in buckets[value]:
                fold[index] = block - 1
        plan = []
        for k in range(splits):
            validation = blocks[k + 1]
            start = ordered.index(validation[0])
            train_values = set(ordered[:max(0, start - gap)])
            train_ids = [rows[index][id_column] for value in ordered if value in train_values for index in buckets[value]]
            if not train_ids:
                raise ValueError("time gap removes all training history")
            plan.append({"fold": k, "train_ids": train_ids,
                         "validation_ids": [rows[i][id_column] for i, value in enumerate(fold) if value == k]})
        return fold, {"strategy": "expanding-time", "gap_unique_times": gap,
                      "plan": plan, "warning": "Use the explicit train_ids, never train on fold != k in time mode."}
    if time_column or gap:
        raise ValueError("--time and --gap only apply to time splits")
    labels = {}
    if target:
        if any(not row[target] for row in rows):
            raise ValueError("empty target")
        labels = {i: row[target] for i, row in enumerate(rows)}
    if regression_bins:
        if not target or regression_bins < 2:
            raise ValueError("regression bins require target and >=2 bins")
        numeric = [(float(row[target]), i) for i, row in enumerate(rows)]
        if any(not math.isfinite(value) for value, _ in numeric):
            raise ValueError("non-finite target")
        unique = sorted(set(value for value, _ in numeric))
        lookup = {value: min(regression_bins - 1, rank * regression_bins // len(unique))
                  for rank, value in enumerate(unique)}
        labels = {i: str(lookup[value]) for value, i in numeric}
    units = defaultdict(list)
    for i, row in enumerate(rows):
        unit = row[group] if group else row[id_column]
        if not unit:
            raise ValueError("empty split unit")
        units[unit].append(i)
    if len(units) < splits:
        raise ValueError("fewer independent units than folds")
    rng = random.Random(seed)
    keys = sorted(units)
    rng.shuffle(keys)
    keys.sort(key=lambda key: -len(units[key]))
    sizes = [0] * splits
    counts = [defaultdict(int) for _ in range(splits)]
    totals = defaultdict(int)
    use_labels = strategy.startswith("stratified")
    if use_labels:
        for label in labels.values():
            totals[label] += 1
    output = [-1] * len(rows)
    for position, key in enumerate(keys):
        members = units[key]
        local = defaultdict(int)
        if use_labels:
            for index in members:
                local[labels[index]] += 1
        def cost(k):
            label_cost = sum((counts[k][label] + count) / totals[label] for label, count in local.items())
            return (label_cost if use_labels else sizes[k], sizes[k], k)
        chosen = position if position < splits else min(range(splits), key=cost)
        sizes[chosen] += len(members)
        for label, count in local.items():
            counts[chosen][label] += count
        for index in members:
            output[index] = chosen
    return output, {"strategy": strategy, "fold_sizes": sizes,
                    "label_counts": [dict(value) for value in counts],
                    "warning": "Grouped stratification is a balance heuristic; inspect class/group coverage."}


def seal(rows, id_column, group, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be in (0,1)")
    ids = [row[id_column] for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("holdout sample IDs must be non-empty and unique")
    unit_column = group or id_column
    units = sorted(set(row[unit_column] for row in rows))
    if not all(units) or len(units) < 2:
        raise ValueError("need at least two non-empty independent units")
    rng = random.Random(seed)
    rng.shuffle(units)
    count = min(len(units) - 1, max(1, round(fraction * len(units))))
    held = set(units[:count])
    return [row for row in rows if row[unit_column] not in held], [row for row in rows if row[unit_column] in held]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    folds = subs.add_parser("folds")
    folds.add_argument("input", type=Path)
    folds.add_argument("output", type=Path)
    folds.add_argument("--strategy", choices=["random", "stratified", "group", "stratified-group", "leave-group-out", "time"], required=True)
    folds.add_argument("--splits", type=int, default=5)
    folds.add_argument("--target")
    folds.add_argument("--time")
    folds.add_argument("--gap", type=int, default=0)
    folds.add_argument("--regression-bins", type=int)
    folds.add_argument("--plan", type=Path, help="required JSON output in time mode")
    holdout = subs.add_parser("holdout")
    holdout.add_argument("input", type=Path)
    holdout.add_argument("development", type=Path)
    holdout.add_argument("sealed", type=Path)
    holdout.add_argument("--fraction", type=float, default=.15)
    for sub in (folds, holdout):
        sub.add_argument("--id", required=True)
        sub.add_argument("--group")
        sub.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    try:
        names, rows = read_rows(args.input)
        if args.action == "folds":
            if args.strategy == "time" and not args.plan:
                raise ValueError("time splits require --plan for explicit training IDs")
            if args.plan and args.output.resolve() == args.plan.resolve():
                raise ValueError("fold CSV and training plan must have distinct output paths")
            if args.output.exists() or args.plan and args.plan.exists():
                raise ValueError("output already exists")
            values, report = assign(rows, args.id, args.strategy, args.splits, args.seed, args.group,
                                    args.target, args.time, args.gap, args.regression_bins)
            if args.plan:
                args.plan.parent.mkdir(parents=True, exist_ok=True)
            write_new(args.output, names + ["fold"], [dict(row, fold=fold) for row, fold in zip(rows, values)])
            if args.plan:
                with args.plan.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2)
        else:
            if args.development.resolve() == args.sealed.resolve() or args.development.exists() or args.sealed.exists():
                raise ValueError("distinct unused output paths required")
            dev, held = seal(rows, args.id, args.group, args.fraction, args.seed)
            write_new(args.development, names, dev)
            write_new(args.sealed, names, held)
            report = {"development_rows": len(dev), "sealed_rows": len(held),
                      "sealed_sha256": hashlib.sha256(args.sealed.read_bytes()).hexdigest()}
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
