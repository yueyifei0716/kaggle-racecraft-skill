"""Finite, weighted metrics for the standalone local analysis commands."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

DIRECTIONS = {"mae": "lower", "rmse": "lower", "accuracy": "higher", "logloss": "lower", "auc": "higher", "qwk": "higher", "f1": "higher"}


def finite(value, label="number"):
    if isinstance(value,bool):
        raise ValueError(f"{label} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def load_scorer(spec):
    """An explicitly supplied local Python file:function executes trusted user code."""
    filename, sep, name = spec.rpartition(":")
    if not sep or not filename or not name:
        raise ValueError("custom scorer must be /path/to/scorer.py:function")
    path = Path(filename)
    module_spec = importlib.util.spec_from_file_location("kaggle_custom_scorer", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("custom scorer must be a local Python file")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    scorer = getattr(module, name)
    if not callable(scorer):
        raise ValueError("custom scorer is not callable")
    return scorer


def score(name, truth, predictions, weights=None, custom=None):
    if not truth or len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same nonzero length")
    y = [finite(v, "y_true") for v in truth]
    p = [finite(v, "y_pred") for v in predictions]
    w = [1.0] * len(y) if weights is None else [finite(v, "weight") for v in weights]
    if len(w) != len(y) or any(v <= 0 for v in w):
        raise ValueError("weights must be positive and match the observations")
    total = sum(w)
    if name == "custom":
        if custom is None:
            raise ValueError("custom metric requires --scorer")
        return finite(custom(y, p, w), "custom score")
    if name == "mae":
        return sum(a * abs(b - c) for a, b, c in zip(w, y, p)) / total
    if name == "rmse":
        return math.sqrt(sum(a * (b - c) ** 2 for a, b, c in zip(w, y, p)) / total)
    if name == "accuracy":
        return sum(a * (b == c) for a, b, c in zip(w, y, p)) / total
    if name == "f1":
        if any(v not in (0,1) for v in y+p):
            raise ValueError("binary F1 requires 0/1 labels and hard predictions; custom scorer supports multiclass")
        tp=sum(a for a,b,c in zip(w,y,p) if b==c==1)
        fp=sum(a for a,b,c in zip(w,y,p) if b==0 and c==1)
        fn=sum(a for a,b,c in zip(w,y,p) if b==1 and c==0)
        return 2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.0
    if name in {"logloss", "auc"}:
        if any(v not in (0, 1) for v in y) or any(v < 0 or v > 1 for v in p):
            raise ValueError("binary metrics require labels 0/1 and probabilities in [0,1]")
        if name == "logloss":
            eps = 1e-15
            return -sum(a * (b * math.log(max(eps, min(1-eps, c))) + (1-b) * math.log(max(eps, min(1-eps, 1-c)))) for a, b, c in zip(w, y, p)) / total
        positives = sum(a for a, b in zip(w, y) if b == 1)
        negatives = total - positives
        if positives == 0 or negatives == 0:
            raise ValueError("AUC requires both classes")
        ranked = sorted(zip(p, y, w))
        numerator, below = 0.0, 0.0
        i = 0
        while i < len(ranked):
            j = i
            pos, neg = 0.0, 0.0
            while j < len(ranked) and ranked[j][0] == ranked[i][0]:
                _, label, weight = ranked[j]
                if label == 1:
                    pos += weight
                else:
                    neg += weight
                j += 1
            numerator += pos * (below + neg / 2)
            below += neg
            i = j
        return numerator / (positives * negatives)
    if name == "qwk":
        if any(v != int(v) for v in y + p):
            raise ValueError("QWK requires integer ordered class labels")
        # Integer distances preserve the ordering and spacing of absent classes.
        truth_hist, pred_hist = {}, {}
        observed = 0.0
        for a, b, c in zip(w, y, p):
            truth_hist[b] = truth_hist.get(b, 0.0) + a
            pred_hist[c] = pred_hist.get(c, 0.0) + a
            observed += a * (b-c) ** 2
        expected = sum(a*b*(c-d)**2 for c,a in truth_hist.items() for d,b in pred_hist.items()) / total
        if expected == 0:
            raise ValueError("QWK is undefined for one constant class")
        return 1 - observed / expected
    raise ValueError(f"unsupported metric: {name}")
