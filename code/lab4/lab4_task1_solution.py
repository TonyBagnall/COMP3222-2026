"""Lab 4, Task 1 solution: distances and nearest neighbours."""

import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_gunpoint


@njit(cache=True)
def euclidean_distance(x, y):
    total = 0.0
    for i in range(len(x)):
        total += (x[i] - y[i]) ** 2
    return np.sqrt(total)


@njit(cache=True)
def dtw_distance(x, y):
    n, m = len(x), len(y)
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            C[i, j] = cost + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    return C[n, m]


class OneNearestNeighbour(ClassifierMixin, BaseEstimator):
    def __init__(self, distance="euclidean"):
        self.distance = distance

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict(self, X):
        distance = dtw_distance if self.distance == "dtw" else euclidean_distance
        return np.array([
            self.y_[np.argmin([distance(x[0], t[0]) for t in self.X_])]
            for x in np.asarray(X, dtype=float)
        ])


if __name__ == "__main__":
    import lab4_task1

    # Use these solutions inside the starter's check function.
    lab4_task1.euclidean_distance = euclidean_distance
    lab4_task1.dtw_distance = dtw_distance
    lab4_task1.check_distances()

    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    perm = np.random.default_rng(0).permutation(X_train.shape[2])
    for name, clf, Xtr, Xte in [
        ("RandomForest", RandomForestClassifier(random_state=0), X_train[:, 0], X_test[:, 0]),
        ("1-NN Euclidean", OneNearestNeighbour("euclidean"), X_train, X_test),
        ("1-NN DTW", OneNearestNeighbour("dtw"), X_train, X_test),
    ]:
        ordered = accuracy_score(y_test, clf.fit(Xtr, y_train).predict(Xte))
        shuffled = accuracy_score(
            y_test, clf.fit(Xtr[..., perm], y_train).predict(Xte[..., perm])
        )
        print(f"{name:15s} ordered {ordered:.3f}  shuffled {shuffled:.3f}")

    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
    print(f"aeon 1-NN DTW   {accuracy_score(y_test, knn.fit(X_train, y_train).predict(X_test)):.3f}")
