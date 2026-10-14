"""Lab 4, Task 2 solution: summary features and random intervals."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_gunpoint
from aeon.transformations.collection.feature_based import Catch22

from lab4_task2 import slope


def summary_features(X):
    return np.concatenate([
        X.mean(axis=2), X.std(axis=2), X.min(axis=2), X.max(axis=2), slope(X)
    ], axis=1)


class SummaryTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return summary_features(np.asarray(X, dtype=float))


class RandomIntervalTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_intervals=10, min_length=3, random_state=None):
        self.n_intervals = n_intervals
        self.min_length = min_length
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        m = np.asarray(X).shape[2]
        self.intervals_ = []
        for _ in range(self.n_intervals):
            start = rng.integers(0, m - self.min_length + 1)
            end = rng.integers(start + self.min_length, m + 1)
            self.intervals_.append((start, end))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        features = []
        for start, end in self.intervals_:
            interval = X[:, :, start:end]
            features += [interval.mean(axis=2), interval.std(axis=2), slope(interval)]
        return np.concatenate(features, axis=1)


if __name__ == "__main__":
    import lab4_task2

    lab4_task2.summary_features = summary_features
    lab4_task2.check_functions()

    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    models = {
        "Summary + RF": make_pipeline(SummaryTransformer(),
                                      RandomForestClassifier(random_state=0)),
        "20 intervals + RF": make_pipeline(RandomIntervalTransformer(20, random_state=0),
                                           RandomForestClassifier(random_state=0)),
        "Catch22 + RF": make_pipeline(Catch22(), RandomForestClassifier(random_state=0)),
        "TimeSeriesForest": TimeSeriesForestClassifier(random_state=0),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name:18s} {accuracy_score(y_test, model.predict(X_test)):.3f}")
