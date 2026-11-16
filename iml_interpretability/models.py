"""Inherently interpretable model candidates.

Extend or replace :func:`get_models` with the models you want to compare, e.g. trees of different
depths to probe the threshold at which a model stops being interpretable, or random forests and
gradient-boosted trees to cover a whole model class.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    return {
        "decision_tree(max_depth=1)": DecisionTreeClassifier(max_depth=1, random_state=seed),
        "decision_tree(max_depth=2)": DecisionTreeClassifier(max_depth=2, random_state=seed),
        "decision_tree(max_depth=3)": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "decision_tree(max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=seed),
        "decision_tree(max_depth=5)": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "decision_tree(max_depth=6)": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "decision_tree(max_depth=8)": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "decision_tree(max_depth=None)": DecisionTreeClassifier(max_depth=None, random_state=seed),
        "random_forest(depth=1,n=3)": RandomForestClassifier(n_estimators=3, max_depth=1, random_state=seed),
        "random_forest(depth=1,n=10)": RandomForestClassifier(n_estimators=10, max_depth=1, random_state=seed),
        "random_forest(depth=1,n=100)": RandomForestClassifier(n_estimators=100, max_depth=1, random_state=seed),
        "random_forest(depth=3,n=3)": RandomForestClassifier(n_estimators=3, max_depth=3, random_state=seed),
        "random_forest(depth=3,n=10)": RandomForestClassifier(n_estimators=10, max_depth=3, random_state=seed),
        "random_forest(depth=3,n=100)": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=seed),
        "random_forest(depth=None,n=100)": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed),
        "gradient_boosting(depth=1,n=3)": GradientBoostingClassifier(n_estimators=3, max_depth=1, random_state=seed),
        "gradient_boosting(depth=1,n=10)": GradientBoostingClassifier(n_estimators=10, max_depth=1, random_state=seed),
        "gradient_boosting(depth=1,n=100)": GradientBoostingClassifier(
            n_estimators=100, max_depth=1, random_state=seed
        ),
        "gradient_boosting(depth=3,n=3)": GradientBoostingClassifier(n_estimators=3, max_depth=3, random_state=seed),
        "gradient_boosting(depth=3,n=10)": GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=seed),
        "gradient_boosting(depth=3,n=100)": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        ),
    }
