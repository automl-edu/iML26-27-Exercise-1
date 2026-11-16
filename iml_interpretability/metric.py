"""Interpretability metrics.

Implement your own metric by subclassing :class:`InterpretabilityMetric`. The metric must be
applicable to every model of the chosen model class (e.g. decision trees, random forests and
gradient-boosted trees), so that scores are comparable across models and datasets.
:class:`RelativeNumberOfLeaves` is a minimal example for a single decision tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset
from iml_interpretability.tree import (
    Node,
    depth,
    effective_n_trees,
    essential_split_share,
    feature_reuse,
    max_features_on_path,
    n_leaves,
    trees_of,
)


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


class RelativeNumberOfLeaves(InterpretabilityMetric):
    """Number of leaves of a decision tree relative to the maximum possible for its depth.

    A binary tree of depth ``d`` has at most ``2 ** d`` leaves, so the score lies in ``(0, 1]``, where
    1 means a complete tree. Lower means the tree uses fewer of the decision rules its depth would
    allow. This is only an example: replace it with your own metric, which should go beyond such
    simple size measures, and argue why it captures interpretability.
    """

    name = "relative_n_leaves"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"{type(self).__name__} only supports DecisionTreeClassifier, got {type(model).__name__}")
        return model.get_n_leaves() / 2 ** model.get_depth()


class StructuralInterpretability(InterpretabilityMetric):
    """
    The weights are not tuned or anything, currently we just have some ideas combined to
    a score with is offset by a decision boundary and put through a sigmoid.

    threshold: so that we can have a level of complexity that marks the point, where sigmoid returns .5
        (could also be removed and complexity could be recentered around zero, but that would mean that negative
        complexity exists which is questionable)

    sigmoid: so we get a probability of being interpretable (class 1) or not (class 0).
        We currently use .5 as decision threshold (higher means more interpretable)!

    weights: one weight per entry of :meth:`features`, matched by name. Pass ``weights`` and/or ``threshold``
        to the constructor to override the defaults below.
    """

    name = "structural_interpretability"

    # fitted on 100 hand labels, see labeling/fit_metric.py
    DEFAULT_TREE_WEIGHTS: dict[str, float] = {
        "depth": 0.31,
        "log2_n_leaves": 1.35,
        "log2_feature_reuse": 0.6,
        "max_features_on_path": 0.36,
        "essential_split_share": 3.84,
    }
    DEFAULT_ENSEMBLE_WEIGHTS: dict[str, float] = {
        "log2_n_trees": 0.97,
        "log2_effective_n_trees": 1.35,
    }
    DEFAULT_THRESHOLD: float = 11.13

    def __init__(
        self,
        tree_weights: dict[str, float] | None = None,
        ensemble_weights: dict[str, float] | None = None,
        threshold: float | None = None,
    ):
        self.tree_weights = dict(self.DEFAULT_TREE_WEIGHTS if tree_weights is None else tree_weights)
        self.ensemble_weights = dict(self.DEFAULT_ENSEMBLE_WEIGHTS if ensemble_weights is None else ensemble_weights)
        self.threshold = self.DEFAULT_THRESHOLD if threshold is None else threshold

    def _tree_features(self, root: Node) -> dict[str, float]:
        """Scaled structural features of the tree, keyed like :attr:`weights`."""
        return {
            "depth": float(depth(root)),
            "log2_feature_reuse": float(np.log2(1 + feature_reuse(root))),
            "max_features_on_path": float(max_features_on_path(root)),
            "log2_n_leaves": float(np.log2(n_leaves(root))),
            "essential_split_share": essential_split_share(root),
        }

    def _tree_complexity(self, root: Node) -> float:
        """Weighted sum of :meth:`features`; 0 for a single leaf, unbounded above."""
        features = self._tree_features(root)

        if features.keys() != self.tree_weights.keys():
            raise ValueError(f"weights {sorted(self.tree_weights)} do not match features {sorted(features)}")
        return float(sum(self.tree_weights[name] * value for name, value in features.items()))

    def _ensemble_features(self, ensemble: list[Node]) -> dict[str, float]:
        return {
            "log2_n_trees": float(np.log2(len(ensemble))),
            "log2_effective_n_trees": float(np.log2(effective_n_trees(ensemble))),
        }

    def _ensemble_complexity(self, ensemble: list[Node]) -> float:
        # log2(1) = 0 for a single tree
        features = self._ensemble_features(ensemble)
        if features.keys() != self.ensemble_weights.keys():
            raise ValueError(f"weights {sorted(self.ensemble_weights)} do not match features {sorted(features)}")
        return float(sum(self.ensemble_weights[name] * value for name, value in features.items()))

    def score(self, model: BaseEstimator, dataset: Dataset):
        """Interpretability score in (0, 1); higher = more interpretable, 0.5 = threshold of interpretability."""
        trees = trees_of(model)
        mean_tc = np.mean([self._tree_complexity(tree) for tree in trees])
        ec = self._ensemble_complexity(trees)
        score = self.threshold - (mean_tc + ec)
        return float(1 / (1 + np.exp(-score)))  # sigmoid
