"""Offline acceptance worker: real small least-squares fits, no model/API calls."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path


HERE = str(Path(__file__).resolve())
STAGES = ["data", "features", "training", "prediction", "scoring"]


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, names, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def basis(row, family):
    x = float(row["x"])
    extra = float(row["a"]) * math.sin(x) if family == "sin" else float(row["b"]) * math.cos(x)
    return [1., x, extra]


def fit(rows, family):
    matrix = [[0.] * 4 for _ in range(3)]
    for row in rows:
        values = basis(row, family)
        for i in range(3):
            for j in range(3):
                matrix[i][j] += values[i] * values[j]
            matrix[i][3] += values[i] * float(row["y_true"])
    for i in range(3):
        pivot = max(range(i, 3), key=lambda k: abs(matrix[k][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        scale = matrix[i][i]
        if abs(scale) < 1e-12:
            raise ValueError("singular synthetic design")
        matrix[i] = [value / scale for value in matrix[i]]
        for j in range(3):
            if j != i:
                scale = matrix[j][i]
                matrix[j] = [value - scale * other for value, other in zip(matrix[j], matrix[i])]
    return [matrix[i][3] for i in range(3)]


def predict(row, family, coefficients):
    return sum(a * b for a, b in zip(basis(row, family), coefficients))


def load_notebook(path):
    notebook = json.loads(Path(path).read_text())
    namespace = {}
    # Only notebooks constructed by this trusted fixture are executed in local tests.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(cell["source"], "synthetic-notebook", "exec"), namespace)
    return namespace


def prepare(request):
    public = request["platform"]["data"]
    rows = read(public["train_path"])
    units = sorted({row["unit"] for row in rows})
    held_units = set(units[-4:])
    development = [row for row in rows if row["unit"] not in held_units]
    holdout = [row for row in rows if row["unit"] in held_units]
    write("development.csv", list(rows[0]), development)
    write("sealed-truth.csv", list(rows[0]), holdout)
    truth_names = ["row_id", "unit", "fold", "group", "y_true"]
    write("validation-truth.csv", truth_names, [{key: row[key] for key in truth_names} for row in development])
    write("holdout-truth.csv", truth_names, [{key: row[key] for key in truth_names} for row in holdout])
    baseline = []
    for row in development:
        history = [other for other in development if other["fold"] != row["fold"]]
        constant = sum(float(other["y_true"]) for other in history) / len(history)
        baseline.append({**{key: row[key] for key in truth_names}, "y_pred": constant})
    write("baseline.csv", truth_names + ["y_pred"], baseline)
    test = read(public["test_path"])
    write("test.csv", list(test[0]), test)
    write("sample.csv", ["id", "prediction"], [{"id": row["id"], "prediction": 0} for row in test])
    Path("sealed-ids.json").write_text(json.dumps(sorted(held_units)))
    print("{}")


def train(identifier, family):
    rows = read("development.csv")
    fold_models = {}
    for fold in sorted({row["fold"] for row in rows}):
        training = [row for row in rows if row["fold"] != fold]
        fold_models[fold] = {"coefficients": fit(training, family), "training_ids": [row["row_id"] for row in training]}
    model = {"family": family, "fold_models": fold_models, "full_coefficients": fit(rows, family),
             "training_ids": [row["row_id"] for row in rows]}
    source = "import math, json\nMODEL = json.loads(" + repr(json.dumps(model)) + ")\n"
    source += "def predict(inputs):\n    out=[]\n    for row in inputs:\n        x=float(row['x'])\n        z=float(row['a'])*math.sin(x) if MODEL['family']=='sin' else float(row['b'])*math.cos(x)\n        out.append(sum(a*b for a,b in zip([1.,x,z],MODEL['full_coefficients'])))\n    return out\n"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
                "cells": [{"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}]}
    Path(identifier + ".ipynb").write_text(json.dumps(notebook))
    names = ["row_id", "unit", "fold", "group", "y_true", "y_pred"]
    write(identifier + "-oof.csv", names,
          [{**{key: row[key] for key in names if key != "y_pred"},
            "y_pred": predict(row, family, fold_models[row["fold"]]["coefficients"])} for row in rows])
    test = read("test.csv")
    predictions = load_notebook(identifier + ".ipynb")["predict"](test)
    write(identifier + "-test.csv", ["id", "prediction"],
          [{"id": row["id"], "prediction": value} for row, value in zip(test, predictions)])
    print("{}")


def validate(request):
    output = request["outputs"]
    namespace = load_notebook(output["artifact"])
    model = namespace["MODEL"]
    dev = read("development.csv")
    held = read("sealed-truth.csv")
    rows = read(output["rows"])
    expected_source = held if request.get("task_kind") == "holdout" else dev
    expected = {row["row_id"]: row for row in expected_source}
    isolated = not ({row["unit"] for row in dev} & {row["unit"] for row in held})
    provenance = set(model["training_ids"]) == {row["row_id"] for row in dev}
    for fold, item in model["fold_models"].items():
        training = [row for row in dev if row["fold"] != fold]
        provenance = provenance and set(item["training_ids"]) == {row["row_id"] for row in training}
        provenance = provenance and all(abs(a-b) < 1e-8 for a, b in zip(item["coefficients"], fit(training, model["family"])))
    prediction_check = set(expected) == {row["row_id"] for row in rows}
    for row in rows:
        original = expected[row["row_id"]]
        coefficients = model["full_coefficients"] if request.get("task_kind") == "holdout" else model["fold_models"][row["fold"]]["coefficients"]
        prediction_check = prediction_check and abs(float(row["y_pred"]) - predict(original, model["family"], coefficients)) < 1e-8
    test = read("test.csv")
    test_values = namespace["predict"](test)
    emitted = read(output.get("predictions", str(Path(output["artifact"]).with_name(Path(output["artifact"]).stem + "-test.csv"))))
    prediction_check = prediction_check and len(test_values) == len(emitted) and all(
        row["id"] == original["id"] and abs(float(row["prediction"])-value) < 1e-8
        for row, original, value in zip(emitted, test, test_values))
    checks = {"prediction_to_artifact": prediction_check, "split_isolation": isolated, "training_provenance": provenance}
    print(json.dumps({"decision": "PASS" if all(checks.values()) else "BLOCK", "checks": checks,
                      "covered_stages": STAGES, "artifact_sha256": hashlib.sha256(Path(output["artifact"]).read_bytes()).hexdigest(),
                      "rows_sha256": hashlib.sha256(Path(output["rows"]).read_bytes()).hexdigest(),
                      "reason": "recomputed grouped-fold training and exact frozen notebook predictions"}))


def holdout():
    held = read("sealed-truth.csv")
    for notebook in sorted(Path(".").glob("candidate-*.ipynb")):
        namespace = load_notebook(notebook)
        values = namespace["predict"](held)
        names = ["row_id", "unit", "fold", "group", "y_true", "y_pred"]
        write(notebook.stem + "-holdout.csv", names,
              [{**{key: row[key] for key in names if key != "y_pred"}, "y_pred": value} for row, value in zip(held, values)])
    print("{}")


def worker(request):
    kind = request["task_kind"]
    with Path("fixture-worker-requests.jsonl").open("a") as stream:
        stream.write(json.dumps({"task_kind": kind, "attempt": request["attempt"], "route": request["route"], "intel": request["intel"]}) + "\n")
    if kind == "bootstrap":
        data = request["platform"]["data"]
        prepared = {**request, "platform": {**request["platform"], "data": {
            **data, "train_path": "inputs/train.csv", "test_path": "inputs/test.csv"}}}
        result = {"kind": "bootstrap", "reason": "discover metric and data from platform; split by independent units",
                  "files": [{"path": "fixture-platform-data.json", "content": json.dumps(prepared)}],
                  "downloads": [{"files": [{"source_path": data["train_path"], "destination": "inputs/train.csv"},
                                             {"source_path": data["test_path"], "destination": "inputs/test.csv"}],
                                 "costs": {"money": 0, "gpu_seconds": 0, "wall_seconds": 4}}],
                  "command": [sys.executable, HERE, "prepare"], "costs": {"wall_seconds": 4, "money": 0, "gpu_seconds": 0},
                  "setup": {"metric": {"name": "mae", "direction": "lower"}, "deadline": data["deadline"],
                            "reserve_hours": .001, "validation_truth": "validation-truth.csv", "baseline_rows": "baseline.csv",
                            "sample_submission": "sample.csv", "holdout_ids": "sealed-ids.json", "holdout_truth": "holdout-truth.csv",
                            "validator_command": [sys.executable, HERE, "validate"], "id_column": "id", "prediction_column": "prediction"}}
    elif kind == "holdout":
        result = {"kind": "holdout", "reason": "evaluate frozen models on sealed units once, no tuning",
                  "files": [], "command": [sys.executable, HERE, "holdout"], "costs": {"wall_seconds": 4, "money": 0, "gpu_seconds": 0},
                  "outputs": {identifier: identifier + "-holdout.csv" for identifier in request["selected_candidates"]}}
    elif request["attempt"] <= 2:
        identifier = "candidate-" + str(request["attempt"])
        family = "sin" if request["attempt"] == 1 else "cos"
        result = {"kind": "experiment", "reason": "fit an independent feature family; latest community evidence is an input",
                  "registration": {"id": identifier, "family": family, "parent": None, "structural": True, "swept": 1,
                                   "hypothesis": "grouped-fold feature family improves constant baseline", "stages": STAGES},
                  "files": [], "command": [sys.executable, HERE, "train", identifier, family], "costs": {"wall_seconds": 4, "money": 0, "gpu_seconds": 0},
                  "outputs": {"artifact": identifier + ".ipynb", "kind": "notebook", "predictions": identifier + "-test.csv", "rows": identifier + "-oof.csv"}}
    else:
        result = {"kind": "stop", "reason": "two distinct validated families available; stop exploration and finalize"}
    print(json.dumps(result))


if __name__ == "__main__":
    mode = sys.argv[1]
    request = json.load(sys.stdin)
    if mode == "worker":
        worker(request)
    elif mode == "prepare":
        # The runner passes execution context; platform data is supplied via a local bootstrap receipt.
        prepare(json.loads(Path("fixture-platform-data.json").read_text()))
    elif mode == "train":
        train(sys.argv[2], sys.argv[3])
    elif mode == "validate":
        validate(request)
    elif mode == "holdout":
        holdout()
    elif mode == "predict":
        print(json.dumps({"predictions": load_notebook(request["artifact_path"])["predict"](request["inputs"])}))
