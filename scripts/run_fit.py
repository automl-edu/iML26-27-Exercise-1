"""Fit the weights and threshold of StructuralInterpretability to the hand labels.

Usage::

    uv run python labeling/fit_metric.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELING_DIR = Path("labeling")
SEED = 0

FEATURES = [
    "depth",
    "log2_n_leaves",
    "log2_feature_reuse",
    "max_features_on_path",
    "essential_split_share",
    "log2_n_trees",
    "log2_effective_n_trees",
]


def load_labeled_items(labeling_dir: Path = LABELING_DIR) -> pd.DataFrame:
    items = pd.read_csv(labeling_dir / "items.csv")
    labels = pd.read_csv(labeling_dir / "labels.csv")

    data = items.merge(labels, on="item_id", how="inner", validate="one_to_one")

    if len(items) - len(data):
        raise ValueError("Check for unlabelled data in dataset!")

    return data


def make_model():
    """Standardise the features, then a logistic regression"""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=20, cv=inner, scoring="neg_log_loss", max_iter=10_000),
    )


def get_weights(model) -> tuple[float, np.ndarray]:
    scaler, regression = model[0], model[-1]
    coef = regression.coef_[0] / scaler.scale_
    intercept = regression.intercept_[0] - coef @ scaler.mean_
    return float(intercept), -coef


# Load data
data = load_labeled_items()
X = data[FEATURES].to_numpy(dtype=float)
y = data["interpretable"].to_numpy(dtype=int)

# Run CV to get an idea of weights and std's
acc, f1, thresholds, weights = [], [], [], []
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
for train, test in outer.split(X, y):
    model = make_model().fit(X[train], y[train])
    pred = model.predict(X[test])
    acc.append(accuracy_score(y[test], pred))
    f1.append(f1_score(y[test], pred, zero_division=0))
    t, w = get_weights(model)
    weights.append(w)
    thresholds.append(t)

weights, thresholds = np.array(weights), np.array(thresholds)
summary = pd.DataFrame({"mean": weights.mean(axis=0), "std": weights.std(axis=0)}, index=FEATURES)
summary.loc["threshold"] = [thresholds.mean(axis=0), thresholds.std(axis=0)]

print(f"Nested CV acc    {np.mean(acc):.2f} ± {np.std(acc):.2f}")
print(f"Nested CV f1     {np.mean(f1):.2f} ± {np.std(f1):.2f}")
print()
print(summary.round(3).to_string())


# Final fit on all labels
final = make_model().fit(X, y)
threshold, w = get_weights(final)
fitted = dict(zip(FEATURES, w, strict=True))
print(f"\nfinal model (C = {final[-1].C_[0]:.3g})")
print(f"threshold                {threshold:.3f}")
for name, value in fitted.items():
    print(f"{name:24s} {value:.3f}")

# Save weights
fitted = {name: round(float(value), 2) for name, value in zip(FEATURES, w, strict=False)}
fitted["threshold"] = round(threshold, 2)
(LABELING_DIR / "fitted_weights.json").write_text(json.dumps(fitted, indent=2) + "\n")
print(json.dumps(fitted, indent=2))
