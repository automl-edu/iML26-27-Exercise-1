"""Tests for the structural tree features in :mod:`iml_interpretability.tree`.

Every feature is checked on the same small hand-built trees, so each expected value can be counted by hand.
The feature functions are looked up as ``tree.<name>`` at test time: until a function exists in ``tree.py``,
its test fails with an ``AttributeError`` naming the missing function.
"""

from __future__ import annotations

import pytest

from iml_interpretability import tree
from iml_interpretability.tree import Node

LEAF = Node()

STUMP = Node(0, LEAF, LEAF)

# imbalanced tree
CHAIN = Node(0, LEAF, Node(1, LEAF, Node(2, LEAF, LEAF)))

# complete tree of depth 3
COMPLETE = Node(0, Node(1, Node(2, LEAF, LEAF), Node(2, LEAF, LEAF)), Node(1, Node(2, LEAF, LEAF), Node(2, LEAF, LEAF)))

# reuses x0
REUSE = Node(0, Node(0, LEAF, LEAF), Node(1, LEAF, Node(0, LEAF, LEAF)))

TREES = {"leaf": LEAF, "stump": STUMP, "chain": CHAIN, "complete": COMPLETE, "reuse": REUSE}


def _cases(expected: dict[str, int]) -> list:
    """Turn ``{tree name: expected value}`` into parametrize cases with readable ids."""
    return [pytest.param(TREES[name], value, id=name) for name, value in expected.items()]


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 1, "stump": 2, "chain": 4, "complete": 8, "reuse": 5}))
def test_n_leaves(node, expected):
    assert tree.n_leaves(node) == expected


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 0, "stump": 1, "chain": 3, "complete": 3, "reuse": 3}))
def test_depth(node, expected):
    assert tree.depth(node) == expected


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 0, "stump": 1, "chain": 3, "complete": 3, "reuse": 2}))
def test_max_features_on_path(node, expected):
    """Largest number of *distinct* features on a single root-to-leaf path."""
    assert tree.max_features_on_path(node) == expected


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 0, "stump": 0, "chain": 0, "complete": 0, "reuse": 2}))
def test_feature_reuse(node, expected):
    """Number of splits on a feature that an ancestor already tested."""
    assert tree.feature_reuse(node) == expected


HEALTHY, SICK = Node(prediction=0), Node(prediction=1)


def test_simplify_merges_split_with_same_prediction():
    assert tree.simplify(Node(0, HEALTHY, HEALTHY)) == HEALTHY


def test_simplify_cascades_through_a_constant_subtree():
    constant = Node(0, Node(1, HEALTHY, HEALTHY), Node(2, Node(3, HEALTHY, HEALTHY), HEALTHY))
    assert tree.simplify(constant) == HEALTHY


def test_simplify_keeps_splits_that_matter():
    mixed = Node(0, Node(1, HEALTHY, HEALTHY), Node(2, HEALTHY, SICK))
    assert tree.simplify(mixed) == Node(0, HEALTHY, Node(2, HEALTHY, SICK))


def test_simplify_never_merges_unknown_predictions():
    assert tree.simplify(COMPLETE) == COMPLETE  # hand-built leaves have prediction None


def test_simplify_merges_equal_leaf_values_of_regression_trees():
    assert tree.simplify(Node(0, Node(prediction=1.91), Node(prediction=1.91))) == Node(prediction=1.91)
    assert tree.simplify(Node(0, Node(prediction=1.91), Node(prediction=1.92))).feature == 0


def _has_redundant_split(node: Node) -> bool:
    if node.is_leaf:
        return False
    both_leaves = node.left.is_leaf and node.right.is_leaf
    if both_leaves and node.left.prediction == node.right.prediction:
        return True
    return _has_redundant_split(node.left) or _has_redundant_split(node.right)


def test_simplify_removes_every_redundant_split_of_an_sklearn_tree():
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_classification(n_samples=500, n_features=5, flip_y=0.2, random_state=0)
    model = DecisionTreeClassifier(max_depth=6, random_state=0).fit(pd.DataFrame(X), y)
    raw = tree.from_sklearn(model)
    simplified = tree.simplify(raw)
    assert _has_redundant_split(raw)  # this tree has some, otherwise the test checks nothing
    assert not _has_redundant_split(simplified)
    assert tree.n_leaves(simplified) < tree.n_leaves(raw) == model.get_n_leaves()


def test_trees_of_returns_trees_as_fitted():
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_classification(n_samples=500, n_features=5, flip_y=0.2, random_state=0)
    model = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
    assert tree.trees_of(model) == [tree.from_sklearn(model)]


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        pytest.param(HEALTHY, 1.0, id="leaf"),
        pytest.param(Node(0, HEALTHY, SICK), 1.0, id="useful_split"),
        pytest.param(Node(0, HEALTHY, HEALTHY), 0.0, id="redundant_split"),
        pytest.param(Node(0, Node(1, HEALTHY, HEALTHY), Node(2, HEALTHY, SICK)), 2 / 3, id="one_of_three_redundant"),
        pytest.param(COMPLETE, 1.0, id="unknown_predictions"),
    ],
)
def test_essential_split_share(node, expected):
    assert tree.essential_split_share(node) == pytest.approx(expected)


def _small(features: list[int]) -> Node:
    """Depth-2 tree: root features[0], children features[1] and features[2], all leaves unknown."""
    return Node(features[0], Node(features[1], LEAF, LEAF), Node(features[2], LEAF, LEAF))


# The three depth-3 trees from the worked example: 10 distinct splits, (7 + 7 + 5) / 3 splits per tree.
TREE_A = Node(1, Node(2, Node(4, LEAF, LEAF), Node(5, LEAF, LEAF)), Node(3, Node(6, LEAF, LEAF), Node(7, LEAF, LEAF)))
TREE_B = Node(1, Node(2, Node(4, LEAF, LEAF), Node(8, LEAF, LEAF)), Node(3, Node(6, LEAF, LEAF), Node(9, LEAF, LEAF)))
TREE_C = Node(1, Node(2, Node(4, LEAF, LEAF), Node(5, LEAF, LEAF)), Node(10, LEAF, LEAF))


def test_split_signatures_use_position_and_feature():
    assert tree.split_signatures(CHAIN) == {("", 0), ("R", 1), ("RR", 2)}
    assert tree.split_signatures(LEAF) == set()


@pytest.mark.parametrize(
    ("trees", "expected"),
    [
        pytest.param([TREE_A], 1.0, id="single_tree"),
        pytest.param([TREE_A, TREE_A, TREE_A], 1.0, id="identical"),
        pytest.param([_small([0, 1, 2]), _small([3, 4, 5])], 2.0, id="nothing_shared"),
        pytest.param([_small([0, 1, 2]), _small([1, 0, 2])], 5 / 3, id="same_features_other_positions"),
        pytest.param([TREE_A, TREE_B, TREE_C], 10 / (19 / 3), id="worked_example"),
        pytest.param([LEAF, LEAF], 1.0, id="only_leaves"),
    ],
)
def test_effective_n_trees(trees, expected):
    assert tree.effective_n_trees(trees) == pytest.approx(expected)


def test_effective_n_trees_lies_between_one_and_n_for_sklearn_ensembles():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    for model in [RandomForestClassifier(n_estimators=10, max_depth=3), GradientBoostingClassifier(n_estimators=10)]:
        trees = tree.trees_of(model.set_params(random_state=0).fit(X, y))
        assert 1.0 <= tree.effective_n_trees(trees) <= len(trees)
