"""Ensembles: a single tree is an ensemble of one, and more trees can only lower the score."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import Dataset
from iml_interpretability.metric import StructuralInterpretability
from iml_interpretability.tree import trees_of


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_single_tree_adds_no_ensemble_complexity(dataset):
    metric = StructuralInterpretability()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    assert metric._ensemble_complexity(trees_of(model)) == 0


@pytest.mark.parametrize(
    "ensemble",
    [
        RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0),
        GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=0),
    ],
    ids=["random_forest", "gradient_boosting"],
)
def test_ensemble_scores_lower_than_single_tree(dataset, ensemble):
    metric = StructuralInterpretability()
    single = DecisionTreeClassifier(max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    ensemble.fit(dataset.X, dataset.y)
    assert len(trees_of(ensemble)) == 10
    assert metric.score(ensemble, dataset) < metric.score(single, dataset)


def test_more_trees_score_lower(dataset):
    metric = StructuralInterpretability()
    scores = []
    for n_trees in [1, 10, 100]:
        forest = RandomForestClassifier(n_estimators=n_trees, max_depth=3, random_state=0).fit(dataset.X, dataset.y)
        scores.append(metric.score(forest, dataset))
    assert scores[0] > scores[1] > scores[2]


def test_rejects_non_tree_models(dataset):
    model = LogisticRegression().fit(dataset.X, dataset.y)
    with pytest.raises(TypeError):
        StructuralInterpretability().score(model, dataset)
