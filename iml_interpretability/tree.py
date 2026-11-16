"""Library-independent tree representation that interpretability metrics operate on."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import BaseDecisionTree


@dataclass(frozen=True)
class Node:
    """A binary tree node; a leaf has no ``feature``.

    :param threshold: Split threshold (``x[feature] <= threshold`` goes left). Not used by the metric, only for display.
    :param prediction: Leaves only: the predicted class (classification trees) or the leaf value (regression trees,
        e.g. the trees inside gradient boosting). ``None`` means unknown, e.g. in hand-built test trees.
    """

    feature: int | None = None
    left: Node | None = None
    right: Node | None = None
    threshold: float | None = None
    prediction: int | float | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def __str__(self):
        return self.render()

    def render(self, depth: int = 0) -> str:
        prefix = "  " * depth
        if self.is_leaf:
            return prefix + ("leaf" if self.prediction is None else f"leaf: {self.prediction}")
        return "\n".join([prefix + f"x{self.feature}", self.left.render(depth + 1), self.right.render(depth + 1)])


def from_sklearn(model: BaseDecisionTree) -> Node:
    if not isinstance(model, BaseDecisionTree):
        raise TypeError(f"Only supports BaseDecisionTree, got {type(model).__name__}")

    tree = model.tree_
    is_classifier = hasattr(model, "classes_")

    def build(node_id: int) -> Node:
        # sklearn marks leaves with -1
        if tree.children_left[node_id] == -1:
            value = tree.value[node_id]
            prediction = int(value.argmax()) if is_classifier else float(value.ravel()[0])
            return Node(prediction=prediction)
        return Node(
            feature=int(tree.feature[node_id]),
            left=build(tree.children_left[node_id]),
            right=build(tree.children_right[node_id]),
            threshold=float(tree.threshold[node_id]),
        )

    return build(0)


def trees_of(model: BaseEstimator) -> list[Node]:
    """All trees of a fitted tree model as fitted; a single tree is an ensemble of one."""
    if isinstance(model, BaseDecisionTree):
        trees = [model]
    elif isinstance(model, BaseEnsemble):
        trees = np.asarray(list(model), dtype=object).ravel()  # need to ravel becuase of GradBoost 2D structure
    else:
        raise TypeError(f"Only supports tree models, got {type(model).__name__}")
    return [from_sklearn(tree) for tree in trees]


def _same_prediction(a: int | float | None, b: int | float | None) -> bool:
    """Whether two leaves predict the same; unknown predictions (``None``) never count as the same."""
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def simplify(node: Node) -> Node:
    """The tree with every split removed that does not change any prediction.

    Works bottom-up: a split whose two children are leaves with the same prediction is replaced by that leaf. Because
    children are simplified first, this cascades, so a whole subtree that always predicts the same collapses into one
    leaf. Predictions of the tree are unchanged. Leaves without a known prediction are never merged.
    """
    if node.is_leaf:
        return node
    left, right = simplify(node.left), simplify(node.right)
    if left.is_leaf and right.is_leaf and _same_prediction(left.prediction, right.prediction):
        return left
    return Node(feature=node.feature, left=left, right=right, threshold=node.threshold)


def essential_split_share(node: Node) -> float:
    """Share of the splits that change at least one prediction, i.e. that survive :func:`simplify`.

    1 means no split is redundant; lower means part of the tree could be removed without changing any prediction.
    A single leaf has no splits and counts as 1.
    """
    n_splits = n_leaves(node) - 1
    if n_splits == 0:
        return 1.0
    return (n_leaves(simplify(node)) - 1) / n_splits


def n_leaves(node: Node) -> int:
    return 1 if node.is_leaf else n_leaves(node.left) + n_leaves(node.right)


def depth(node: Node) -> int:
    return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))


def max_features_on_path(node: Node, seen: frozenset[int] = frozenset()) -> int:
    """Largest number of distinct features tested on a single root-to-leaf path.

    :param seen: Features already tested above ``node``; leave empty when calling on the root.
    """
    if node.is_leaf:
        return len(seen)
    seen = seen | {node.feature}
    return max(max_features_on_path(node.left, seen), max_features_on_path(node.right, seen))


def feature_reuse(node: Node, seen: frozenset[int] = frozenset()) -> int:
    """Number of splits on a feature that an ancestor of the split already tested.

    :param seen: Features already tested above ``node``; leave empty when calling on the root.
    """
    if node.is_leaf:
        return 0
    reused = int(node.feature in seen)
    seen = seen | {node.feature}
    return reused + feature_reuse(node.left, seen) + feature_reuse(node.right, seen)


def split_signatures(node: Node, path: str = "") -> set[tuple[str, int]]:
    """Every split of a tree as (position, feature); thresholds are ignored.

    The position is the path from the root, e.g. ``""`` for the root and ``"LR"`` for the right child of the root's
    left child, so the same feature tested at the same place in two trees gives the same signature.
    """
    if node.is_leaf:
        return set()
    return {(path, node.feature)} | split_signatures(node.left, path + "L") | split_signatures(node.right, path + "R")


def effective_n_trees(trees: list[Node]) -> float:
    """How many different trees an ensemble is worth: distinct split signatures / average splits per tree.

    1 for a single tree or identical trees, ``len(trees)`` for trees that share no split, in between otherwise.
    Only-leaf ensembles count as 1.
    """
    per_tree = [split_signatures(tree) for tree in trees]
    mean_splits = float(np.mean([len(signatures) for signatures in per_tree]))
    if mean_splits == 0:
        return 1.0
    return len(set().union(*per_tree)) / mean_splits
