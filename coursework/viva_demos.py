"""Short demos for the Assignment 2 viva.

Put this file in the folder that contains your solution/ folder (next to part3_setup.py) and
run the demo the lab assistant asks for, for example:

    python viva_demos.py demo_1

Each demo takes a few seconds. You will be asked to explain the output using your own code.
"""

import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from solution.ensemble_selection import GreedyEnsembleSelector
from solution.pool import generate_oof_probas


def iris_pool():
    return [("stump", DecisionTreeClassifier(max_depth=1, random_state=0)),
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("lr", LogisticRegression(max_iter=500))]


def fitted(sel, name):
    """A fitted attribute, with or without the trailing underscore."""
    return getattr(sel, name + "_", getattr(sel, name, "(missing)"))


def fit_selector(pool, X, y, metric="accuracy", **kw):
    """Fit with the metric name the brief uses; fall back to 'log loss' if that is the spelling used."""
    try:
        return GreedyEnsembleSelector(pool, metric=metric, random_state=0, **kw).fit(X, y)
    except ValueError:
        if metric != "log_loss":
            raise
        return GreedyEnsembleSelector(pool, metric="log loss", random_state=0, **kw).fit(X, y)


def show(label, sel):
    scores = [round(float(s), 4) for s in fitted(sel, "selection_scores")]
    print(f"{label:9s} selected {fitted(sel, 'selected_names')}  scores {scores}")


def demo_1():
    """Accuracy against log loss on iris."""
    X, y = load_iris(return_X_y=True)
    for metric in ("accuracy", "log_loss"):
        show(metric, fit_selector(iris_pool(), X, y, metric))


def demo_2():
    """Which class wins a tie in predict?"""
    X = np.zeros((6, 1))
    y = np.array(["b", "a", "b", "a", "b", "a"])
    sel = GreedyEnsembleSelector([("d", DummyClassifier())], cv=2, random_state=0).fit(X, y)
    print("training labels  ", y)
    print("predict_proba    ", sel.predict_proba(X[:1]))
    print("predict          ", sel.predict(X[:1]))


def demo_3():
    """Fitting the same selector twice."""
    X, y = load_iris(return_X_y=True)
    sel = GreedyEnsembleSelector(iris_pool(), random_state=0)
    sel.fit(X, y)
    show("first fit", sel)
    print("estimators after the first fit:", sel.get_params()["estimators"])
    sel.fit(X, y)
    show("second", sel)


def demo_4():
    """The first member chosen, against each classifier's out-of-fold accuracy."""
    X, y = load_iris(return_X_y=True)
    _, scores = generate_oof_probas(iris_pool(), X, y, cv=5, random_state=0)
    print("out-of-fold accuracy:", {n: round(float(s), 4) for n, s in zip(["stump", "tree", "lr"], scores)})
    for metric in ("accuracy", "log_loss"):
        show(metric, fit_selector(iris_pool(), X, y, metric))


def demo_5():
    """max_ensemble_size limiting the selection."""
    X, y = load_iris(return_X_y=True)
    for size in (None, 1):
        show(f"size {size}", fit_selector(iris_pool(), X, y, "log_loss", max_ensemble_size=size))


def demo_6():
    """A class with a single case: the fold that holds it out cannot learn it."""
    X = np.arange(11, dtype=float).reshape(-1, 1)
    y = np.array(["a"] * 5 + ["b"] * 5 + ["c"])
    oof, scores = generate_oof_probas([("tree", DecisionTreeClassifier(random_state=0))], X, y,
                                      cv=5, random_state=0)
    print("classes           ", np.unique(y))
    print("out-of-fold row for the 'c' case:", np.round(oof[0, 10], 3))
    print("out-of-fold accuracy:", round(float(scores[0]), 4))


def demo_7():
    """String labels: which column is which class?"""
    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    y = np.array(["zebra", "zebra", "zebra", "ant", "ant", "ant"])
    sel = GreedyEnsembleSelector([("lr", LogisticRegression())], cv=3, random_state=0).fit(X, y)
    print("classes_         ", fitted(sel, "classes"))
    print("predict_proba x=0", np.round(sel.predict_proba(np.array([[0.0]])), 3))
    print("predict x=0      ", sel.predict(np.array([[0.0]])))


def demo_8():
    """Two identical classifiers in the pool."""
    X, y = load_iris(return_X_y=True)
    pool = [("lr_a", LogisticRegression(max_iter=500)), ("lr_b", LogisticRegression(max_iter=500)),
            ("stump", DecisionTreeClassifier(max_depth=1, random_state=0))]
    show("accuracy", fit_selector(pool, X, y, "accuracy"))


DEMOS = {f"demo_{i}": f for i, f in enumerate(
    [demo_1, demo_2, demo_3, demo_4, demo_5, demo_6, demo_7, demo_8], 1)}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) == 2 else ""
    name = name if name.startswith("demo_") else f"demo_{name}"
    if name not in DEMOS:
        sys.exit("usage: python viva_demos.py demo_N   (one of " + ", ".join(DEMOS) + ")")
    DEMOS[name]()
