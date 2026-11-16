"""Tests basic requirements for the interpretability metric
on synthetic data and synthetic trees
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import Dataset
from iml_interpretability.metric import StructuralInterpretability
from iml_interpretability.tree import Node, depth, from_sklearn, n_leaves


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def _split_features_preorder(node: Node) -> list[int]:
    """Split features in pre-order: node, then left subtree, then right subtree."""
    if node.is_leaf:
        return []
    return [node.feature] + _split_features_preorder(node.left) + _split_features_preorder(node.right)


def _complete_tree(depth: int, feature: int = 0) -> Node:
    """Every path has length ``depth``; each level splits on its own feature."""
    if depth == 0:
        return Node()
    return Node(feature, _complete_tree(depth - 1, feature + 1), _complete_tree(depth - 1, feature + 1))


@pytest.mark.parametrize("max_depth", [1, 2, 3, None])
def test_from_sklearn_matches_fitted_tree(dataset, max_depth):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(dataset.X, dataset.y)
    node = from_sklearn(model)

    # print(f"\n--- max_depth={max_depth}: sklearn ---")
    # print(export_text(model, feature_names=dataset.feature_names))
    # print(f"--- max_depth={max_depth}: Node ---")
    # print(node)

    assert n_leaves(node) == model.get_n_leaves()
    assert depth(node) == model.get_depth()
    assert _split_features_preorder(node) == [int(f) for f in model.tree_.feature if f >= 0]


def test_same_score_for_invariant_sklearn_trees(dataset):
    metric = StructuralInterpretability()

    # Renaming columns: same data under new names grows the same tree, so the score must not change.
    renamed = Dataset(name=dataset.name, X=dataset.X.add_prefix("renamed_"), y=dataset.y)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    model_renamed = DecisionTreeClassifier(max_depth=3, random_state=0).fit(renamed.X, renamed.y)
    print(from_sklearn(model))
    print(from_sklearn(model_renamed))
    assert from_sklearn(model) == from_sklearn(model_renamed)
    assert metric.score(model, dataset) == metric.score(model_renamed, renamed)


def test_same_score_for_invariant_node_trees():
    metric = StructuralInterpretability()
    # Relabelling feature by hand (0 -> 3, 1 -> 4)
    leaf = Node()
    tree = Node(0, Node(1, leaf, leaf), Node(0, leaf, leaf))
    relabelled = Node(3, Node(4, leaf, leaf), Node(3, leaf, leaf))
    assert metric._tree_complexity(tree) == metric._tree_complexity(relabelled)


def test_score_mirrored_node_trees():
    metric = StructuralInterpretability()
    # Swapping left and right children at every node: which branch is "left" must not matter.
    leaf = Node()
    original = Node(0, Node(1, leaf, Node(2, leaf, leaf)), leaf)
    mirrored = Node(0, leaf, Node(1, Node(2, leaf, leaf), leaf))
    assert original != mirrored
    assert metric._tree_complexity(original) == metric._tree_complexity(mirrored)


def test_score_near_one_for_stump_sklearn_tree(dataset):
    metric = StructuralInterpretability()
    # model with depth 1
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    assert 0.95 < metric.score(model, dataset) <= 1


def test_fully_grown_sklearn_tree_scores_lower():
    metric = StructuralInterpretability()
    X, y = make_classification(n_samples=300, n_features=5, flip_y=0.2, random_state=0)
    noisy_dataset = Dataset(name="noisy", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))

    scores = {}
    for max_depth in [1, 3, None]:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(noisy_dataset.X, noisy_dataset.y)
        scores[max_depth] = metric.score(model, noisy_dataset)

    assert scores[1] > scores[3] > scores[None]


def test_complete_node_tree_scores_lower_than_stump():
    metric = StructuralInterpretability()
    stump = _complete_tree(1)
    complete = _complete_tree(6)  # 64 leaves; RelativeNumberOfLeaves rates this 1.0, the same as the stump
    assert metric._tree_complexity(complete) > metric._tree_complexity(stump)


def test_imbalanced_node_tree_scores_higher_than_complete_tree_of_same_depth():
    metric = StructuralInterpretability()
    leaf = Node()
    # Both have depth 3 and use features 0, 1, 2, but the chain needs 3 decisions instead of 7.
    balanced = _complete_tree(3)
    imbalaned = Node(0, leaf, Node(1, leaf, Node(2, leaf, leaf)))
    assert depth(imbalaned) == depth(balanced)
    assert metric._tree_complexity(imbalaned) < metric._tree_complexity(balanced)
