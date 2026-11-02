"""Lab 5, Task 1: comparing classifiers on a single problem (student starter)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from aeon.classification.sklearn import RotationForestClassifier

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_breast_cancer_data():
    """Return X, y and class names, with malignant as the positive class 1.

    The scikit-learn loader codes malignant as 0 and benign as 1, so the labels
    are recoded here. class_names[label] gives the name of each label.
    """
    data = load_breast_cancer()
    y = 1 - data.target
    return data.data, y, ["benign", "malignant"]


def load_iris_data():
    """Return X, y and class names for Iris (labels 0, 1 and 2)."""
    data = load_iris()
    return data.data, data.target, list(data.target_names)


def make_classifiers(random_state=RANDOM_STATE):
    """Return the three classifiers to compare, keyed by a short name.

    Put preprocessing and your classifier in a Pipeline if it needs any, so the
    preprocessing is fitted on training data only. bakeoff.py reuses this function.
    """
    return {
        "RandF": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "RotF": RotationForestClassifier(n_estimators=100, random_state=random_state),
        # TODO: Replace this stand-in with a classifier you implemented, such as
        #       your BasicEnsembleClassifier from Lab 3.
        "Yours": DecisionTreeClassifier(random_state=random_state),
    }


def single_split(y, random_state=RANDOM_STATE):
    """Return train and test row indices for a stratified 70/30 split.

    The row indices are the case identifiers to save with your predictions.
    """
    ids = np.arange(len(y))
    return train_test_split(ids, test_size=0.3, stratify=y, random_state=random_state)


def make_folds(random_state=RANDOM_STATE):
    """The 5-fold stratified cross-validation used in 1.2 and the nested extension."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def class_column(model, label):
    """Return the predict_proba column of label, using the order in model.classes_."""
    return int(np.flatnonzero(model.classes_ == label)[0])


def labelled_confusion_matrix(y_true, y_pred, class_names):
    """Return the confusion matrix as a DataFrame with actual classes as rows."""
    labels = np.arange(len(class_names))
    counts = pd.crosstab(pd.Categorical(y_true, categories=labels),
                         pd.Categorical(y_pred, categories=labels), dropna=False)
    counts.index = [f"actual {name}" for name in class_names]
    counts.columns = [f"predicted {name}" for name in class_names]
    return counts


def save_predictions(path, ids, y_true, y_pred, proba, class_names):
    """Save case ids, true labels, predictions and class probabilities to a CSV."""
    table = pd.DataFrame({"case_id": ids, "true": y_true, "predicted": y_pred})
    for k, name in enumerate(class_names):
        table[f"p_{name}"] = proba[:, k]
    table.to_csv(path, index=False)


def compute_metrics(y_true, y_pred, proba):
    """Return a dict of the 1.1 measures for one classifier.

    proba has one column per class, in label order 0, 1, ... For both problems
    return accuracy, balanced accuracy and log loss. With two classes, also return
    precision, recall, F1 and AUROC for the positive class 1. With more classes,
    return macro-averaged precision, recall and F1 and macro one-versus-rest AUROC.
    See L5.1 slides 31 and 32.
    """
    raise NotImplementedError("Implement compute_metrics")


def check_metrics():
    """Compare compute_metrics with hand-worked values for four cases."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    proba = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
    expected = {"accuracy": 0.75, "balanced_accuracy": 0.75, "precision": 1.0,
                "recall": 0.5, "f1": 0.667, "auroc": 1.0, "log_loss": 0.400}
    found = compute_metrics(y_true, y_pred, proba)
    for name, value in expected.items():
        print(f"{name:18s} expected {value:.3f}  yours {found.get(name, float('nan')):.3f}")


def cross_val_predictions(model, X, y, folds):
    """Return out-of-fold predictions and probabilities for every case.

    For each fold, clone the model, fit it on the training rows and predict the
    held-out rows, so no case is predicted by a model that saw it. Store the
    probabilities in label order, using class_column to find each column.
    """
    raise NotImplementedError("Implement cross_val_predictions")


# For 1.3. Keys must match make_classifiers. If a classifier is in a Pipeline,
# prefix each parameter with the step name, for example "randomforestclassifier__max_depth".
PARAM_GRIDS = {
    "RandF": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]},
    # TODO: Add a similarly small grid for a second classifier.
}


def tune(model, param_grid, X_train, y_train, random_state=RANDOM_STATE):
    """Return a fitted GridSearchCV that selects parameters by accuracy.

    Use 5-fold stratified cross-validation on the training data only. With the
    default refit=True, the best configuration is refitted on all training data.
    """
    raise NotImplementedError("Implement tune")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    problems = {"breast_cancer": load_breast_cancer_data(), "iris": load_iris_data()}
    for name, (X, y, class_names) in problems.items():
        counts = ", ".join(f"{c} {n}" for c, n in zip(class_names, np.bincount(y)))
        train_ids, test_ids = single_split(y)
        print(f"{name}: X {X.shape}, classes: {counts}")
        print(f"  split seed {RANDOM_STATE}: {len(train_ids)} train, {len(test_ids)} test")
    print(f"Classifiers: {', '.join(make_classifiers())}")
    print(f"Save your results in {RESULTS_DIR}")

    # TODO 1.1: Implement compute_metrics, then call check_metrics().
    # TODO 1.1: For each problem, fit every classifier on the same training rows.
    #       Report the measures and a labelled confusion matrix on the test rows.
    #       Save a results CSV, and the predictions with save_predictions.
    # TODO 1.2: Implement cross_val_predictions. Use one make_folds() for every
    #       classifier, compute the measures from the pooled out-of-fold results,
    #       and plot the three breast cancer ROC curves on one set of axes.
    # TODO 1.3: Implement tune and add a second grid. Tune two classifiers on the
    #       1.1 training rows, report best_params_ and fitting time, and compare
    #       with the untuned test results.
