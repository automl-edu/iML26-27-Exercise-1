"""Build a set of trees and small ensembles for hand-labeling interpretability.

Fits many differently shaped trees on the project datasets and picks a varied subset of 40, adds
30 small random forests and 30 small boosted models (2-8 trees of depth 1-3), and writes

- ``labeling/items.csv``: one row per item with the features the metric uses (averaged over the
  trees of an item) and the score under the current default weights,
- ``labeling/label_trees.html``: a page that shows the items in random order, without any score,
  and exports ``labels.csv`` with one yes/no label per item.

The selection covers the whole complexity range evenly (equal-width bands), and inside each band it
picks items that differ as much as possible in their features,
so that depth, leaves, features per path and reuse do not all move together.

Usage::

    uv run python labeling/make_labeling_set.py
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from iml_interpretability.data import Dataset, load_diabetes, load_endgames, load_titanic, load_wdbc  # noqa: E402
from iml_interpretability.metric import StructuralInterpretability  # noqa: E402
from iml_interpretability.tree import from_sklearn  # noqa: E402

OUTPUT_DIR = Path("labeling")
SEED = 0
MAX_LEAVES = 48  # wider trees do not fit on a screen

# Complexity bands (under the current default weights) and how many single trees to take from each.
# Denser around the current threshold of 6, where labels are most informative for the fit.
N_BANDS = 8  # equal-width complexity bands, so no part of the range is favoured
PER_BAND = 5  # 8 bands x 5 = 40 single trees, balanced against 30 forests and 30 boosted models

# Every feature the metric has is exported to items.csv, so new metric terms show up there automatically.
TREE_FEATURES = list(StructuralInterpretability.DEFAULT_TREE_WEIGHTS)
ENSEMBLE_FEATURES = list(StructuralInterpretability.DEFAULT_ENSEMBLE_WEIGHTS)
# The selection of items only looks at these, so adding metric terms does not change which items are labeled.
# Fixed weights for sorting candidates into complexity bands. Frozen on purpose: changing the metric's weights
# (e.g. after fitting) must not change which items are in the labeling set.
SELECTION_WEIGHTS = {"depth": 0.5, "log2_feature_reuse": 0.2, "max_features_on_path": 0.5, "log2_n_leaves": 0.8}
SELECTION_FEATURES = list(SELECTION_WEIGHTS)

CLASS_NAMES = {
    "titanic": ["died", "survived"],
    "diabetes": ["healthy", "diabetes"],
    "wdbc": ["benign", "malignant"],
    "kr-vs-kp": ["no win", "win"],
}

QUESTION = (
    "Could you explain how this model makes its predictions to a colleague in about a minute, "
    "and follow any single prediction through it by hand? "
    "Judge the structure: assume every feature name is meaningful to you (wdbc's V1 to V30 are cell measurements)."
)


def estimators_of(model) -> list:
    """The sklearn trees of a fitted tree model; a single tree is an ensemble of one."""
    if isinstance(model, DecisionTreeClassifier):
        return [model]
    return list(np.asarray(model.estimators_, dtype=object).ravel())


def item_features(estimators: list) -> dict[str, float]:
    """The metric's own features, averaged over the trees, plus the ensemble features."""
    metric = StructuralInterpretability()
    nodes = [from_sklearn(estimator) for estimator in estimators]
    per_tree = pd.DataFrame([metric._tree_features(node) for node in nodes])
    features = per_tree.mean().to_dict()
    features.update(metric._ensemble_features(nodes))
    features["n_trees"] = len(nodes)
    features["mean_n_leaves"] = float(np.mean([estimator.get_n_leaves() for estimator in estimators]))
    features["current_complexity"] = float(
        sum(metric.tree_weights[k] * features[k] for k in TREE_FEATURES)
        + sum(metric.ensemble_weights[k] * features[k] for k in ENSEMBLE_FEATURES)
    )
    features["current_score"] = float(1 / (1 + np.exp(-(metric.threshold - features["current_complexity"]))))
    return features


def single_tree_candidates(datasets: list[Dataset]) -> pd.DataFrame:
    """Many differently shaped single trees: depth, leaf budget and leaf size varied, plus random-split trees."""
    rows = []
    for dataset in datasets:
        configs = [
            {"max_depth": d, "max_leaf_nodes": leaves, "min_samples_leaf": m}
            for d in [1, 2, 3, 4, 5, 6, 8, None]
            for leaves in [None, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]
            for m in [1, 10, 40]
        ]
        # Random split features give unusual shapes (lopsided, repeated features) that greedy trees rarely produce.
        configs += [{"max_depth": d, "max_features": 1, "seed": s} for d in [3, 4, 5, 6] for s in range(5)]
        for config in configs:
            params = {k: v for k, v in config.items() if k != "seed"}
            model = DecisionTreeClassifier(random_state=config.get("seed", SEED), **params).fit(dataset.X, dataset.y)
            if model.get_n_leaves() < 2 or model.get_n_leaves() > MAX_LEAVES:
                continue
            rows.append({"dataset": dataset.name, "kind": "tree", "config": json.dumps(config), "model": model})
    candidates = pd.DataFrame(rows)
    features = pd.DataFrame([item_features([model]) for model in candidates["model"]], index=candidates.index)
    candidates = pd.concat([candidates, features], axis=1)
    # Identical feature vectors carry no new information for the fit.
    return candidates.drop_duplicates(subset=["dataset", *SELECTION_FEATURES]).reset_index(drop=True)


def spread_out(candidates: pd.DataFrame, k: int, rng: np.random.Generator) -> list[int]:
    """Pick ``k`` rows that are far apart in feature space (and across datasets): farthest-point sampling."""
    if len(candidates) <= k:
        return list(candidates.index)
    numeric = candidates[SELECTION_FEATURES].to_numpy(dtype=float)
    numeric = (numeric - numeric.mean(axis=0)) / (numeric.std(axis=0) + 1e-9)
    points = np.hstack([numeric, pd.get_dummies(candidates["dataset"]).to_numpy(dtype=float)])
    chosen = [int(rng.integers(len(points)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(distance.argmax())
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(points - points[nxt], axis=1))
    return list(candidates.index[chosen])


def select_single_trees(candidates: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    selection_complexity = sum(weight * candidates[name] for name, weight in SELECTION_WEIGHTS.items())
    band = pd.cut(selection_complexity, N_BANDS, labels=False)
    chosen: list[int] = []
    for b in range(N_BANDS):
        chosen += spread_out(candidates[band == b], PER_BAND, rng)
    missing = PER_BAND * N_BANDS - len(chosen)
    if missing > 0:  # a sparse band: top up with the most different remaining trees
        chosen += spread_out(candidates.drop(index=chosen), missing, rng)
    return candidates.loc[chosen]


def ensemble_items(datasets: list[Dataset]) -> pd.DataFrame:
    """Small forests and boosted models: 2, 3, 4, 5 or 8 trees of depth 1, 2 or 3, each on two datasets."""
    rows = []
    i = 0
    for family, cls in [("forest", RandomForestClassifier), ("boosting", GradientBoostingClassifier)]:
        for n_trees in [2, 3, 4, 5, 8]:
            for max_depth in [1, 2, 3]:
                for _ in range(2):  # every configuration on two different datasets
                    dataset = datasets[i % len(datasets)]
                    i += 1
                    model = cls(n_estimators=n_trees, max_depth=max_depth, random_state=SEED)
                    model.fit(dataset.X, dataset.y)
                    config = {"n_estimators": n_trees, "max_depth": max_depth}
                    rows.append(
                        {"dataset": dataset.name, "kind": family, "config": json.dumps(config), "model": model}
                        | item_features(estimators_of(model))
                    )
    return pd.DataFrame(rows)


def _draw_tree(ax: plt.Axes, estimator, feature_names: list[str], class_names: list[str] | None) -> None:
    """Draw one fitted sklearn tree: split conditions in the inner nodes, predictions in the leaves."""
    tree = estimator.tree_
    positions: dict[int, tuple[float, float]] = {}
    next_leaf = [0.0]

    def place(node: int, level: int) -> float:
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            x = next_leaf[0]
            next_leaf[0] += 1
        else:
            x = (place(left, level + 1) + place(right, level + 1)) / 2
        positions[node] = (x, -level)
        return x

    place(0, 0)
    for node, (x, y) in positions.items():
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            value = tree.value[node].ravel()
            text = class_names[int(value.argmax())] if class_names is not None else f"{value[0]:+.2f}"
            box = {"boxstyle": "round,pad=0.3", "facecolor": "#e8f1fb", "edgecolor": "#2a78d6"}
        else:
            for child, edge in [(left, "yes"), (right, "no")]:
                cx, cy = positions[child]
                ax.plot([x, cx], [y, cy], color="#8a8980", linewidth=0.8, zorder=1)
                ax.text((x + cx) / 2, (y + cy) / 2, edge, fontsize=6, color="#8a8980", ha="center", va="center")
            text = f"{feature_names[tree.feature[node]]} ≤ {tree.threshold[node]:.3g}"
            box = {"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#3d3d3a"}
        ax.text(x, y, text, ha="center", va="center", fontsize=7, bbox=box, zorder=2)
    ax.set_xlim(-0.7, max(next_leaf[0] - 0.3, 0.7))
    ax.set_ylim(-estimator.get_depth() - 0.5, 0.5)
    ax.axis("off")


def render(item: pd.Series, dataset: Dataset) -> str:
    """Render all trees of an item side by side and return the PNG as a data URI."""
    estimators = estimators_of(item["model"])
    is_boosting = item["kind"] == "boosting"  # boosting trees are regressors on the log-odds scale
    class_names = None if is_boosting else CLASS_NAMES[dataset.name]
    widths = [max(2.5, 1.05 * e.get_n_leaves()) for e in estimators]
    height = 0.85 * (max(e.get_depth() for e in estimators) + 1) + 0.4
    fig, axes = plt.subplots(1, len(estimators), figsize=(sum(widths), height), width_ratios=widths, squeeze=False)
    for ax, estimator in zip(axes[0], estimators, strict=True):
        _draw_tree(ax, estimator, dataset.feature_names, class_names)
    if is_boosting:
        fig.suptitle("prediction = sum of the leaf values of all trees (> 0 means positive class)", fontsize=8)
    elif len(estimators) > 1:
        fig.suptitle("prediction = majority vote of all trees", fontsize=8)
    buffer = io.BytesIO()
    dpi = min(110, 6000 / sum(widths))  # keep very wide ensembles below ~6000 px
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# The labeling page: plain HTML/JS with __QUESTION__ and __ITEMS__ placeholders.
PAGE_TEMPLATE = Path(__file__).with_name("label_page_template.html")


def main() -> None:
    rng = np.random.default_rng(SEED)
    datasets = [load_titanic(), load_diabetes(), load_wdbc(), load_endgames()]
    by_name = {dataset.name: dataset for dataset in datasets}

    singles = select_single_trees(single_tree_candidates(datasets), rng)
    items = pd.concat([singles, ensemble_items(datasets)], ignore_index=True)
    items = items.sample(frac=1, random_state=SEED).reset_index(drop=True)  # random order hides complexity
    items["item_id"] = [f"item_{k + 1:02d}" for k in range(len(items))]

    page_items = [
        {"id": row["item_id"], "dataset": row["dataset"], "img": render(row, by_name[row["dataset"]])}
        for _, row in items.iterrows()
    ]
    OUTPUT_DIR.mkdir(exist_ok=True)
    page = PAGE_TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("__QUESTION__", QUESTION).replace("__ITEMS__", json.dumps(page_items))
    (OUTPUT_DIR / "label_trees.html").write_text(page, encoding="utf-8")

    columns = ["item_id", "dataset", "kind", "config", "n_trees", *TREE_FEATURES, *ENSEMBLE_FEATURES]
    columns += ["mean_n_leaves", "current_complexity", "current_score"]
    items[columns].to_csv(OUTPUT_DIR / "items.csv", index=False)

    print(items.groupby("kind").size().to_string())
    print(items[["current_score"]].describe().round(3).to_string())
    print(f"Wrote {OUTPUT_DIR / 'label_trees.html'} and {OUTPUT_DIR / 'items.csv'}")


if __name__ == "__main__":
    main()
