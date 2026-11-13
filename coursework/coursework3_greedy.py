"""Reference greedy ensemble selection for COMP3222 coursework 3.

This is the forward selection algorithm from the Ensemble Selection
coursework, adapted to work on precomputed LOOCV predictions of 1-NN
classifiers. Because a 1-NN classifier gives one-hot probabilities,
averaging probabilities is the same as an equal-weight vote, so the
ensemble prediction is a majority vote and the score is accuracy.

The rules are the same as in the Ensemble Selection coursework:

* start with the candidate with the best individual score, taking the first
  in candidate order if several are equal;
* at each step, add the candidate that gives the enlarged ensemble the best
  score, taking the first in candidate order if several are equal;
* never select a candidate twice;
* stop when no addition gives a strict improvement, or when every candidate
  has been selected.

Ties in the vote go to the class that comes first in ``classes``.

Example::

    from coursework3_greedy import greedy_select

    # loocv_preds has one row per candidate, one column per training case
    selected, scores = greedy_select(loocv_preds, y_train, np.unique(y_train))
"""

import numpy as np


def _class_indices(predictions, classes):
    """Map labels to their positions in classes, checking they all appear."""
    lookup = {c: i for i, c in enumerate(classes.tolist())}
    try:
        return np.vectorize(lookup.__getitem__, otypes=[np.intp])(predictions)
    except KeyError as err:
        raise ValueError(f"Prediction {err} is not in classes.") from None


def _n_correct(votes, y_index):
    """Number of cases where the vote winner is the true class.

    np.argmax returns the first maximum, so tied votes go to the class that
    comes first in classes.
    """
    return int(np.sum(np.argmax(votes, axis=1) == y_index))


def greedy_select(predictions, y, classes):
    """Select a subset of candidates by greedy forward selection.

    Parameters
    ----------
    predictions : array-like of shape (n_candidates, n_cases)
        The LOOCV predicted class labels of each candidate on the training
        cases.
    y : array-like of shape (n_cases,)
        The true class labels of the training cases.
    classes : array-like of shape (n_classes,)
        The class labels in tie-breaking order, normally ``np.unique(y)``.

    Returns
    -------
    selected : list of int
        The indices of the selected candidates, in the order they were added.
    scores : list of float
        ``scores[i]`` is the LOOCV accuracy of the equal-weight vote of the
        first ``i + 1`` selected candidates.
    """
    classes = np.asarray(classes)
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[0] == 0:
        raise ValueError("predictions must have shape (n_candidates, n_cases).")
    if predictions.shape[1] != len(y):
        raise ValueError("predictions and y have different numbers of cases.")

    n_candidates, n_cases = predictions.shape
    pred_index = _class_indices(predictions, classes)
    y_index = _class_indices(np.asarray(y), classes)

    # one_hot[j] holds the one-hot votes of candidate j, shape (n_cases, n_classes)
    one_hot = np.zeros((n_candidates, n_cases, len(classes)), dtype=np.intp)
    one_hot[np.arange(n_candidates)[:, None], np.arange(n_cases), pred_index] = 1

    # Compare counts of correct cases rather than float accuracies, so that
    # "strict improvement" is exact.
    individual = [_n_correct(one_hot[j], y_index) for j in range(n_candidates)]
    first = int(np.argmax(individual))  # first maximum on ties
    selected = [first]
    best = individual[first]
    scores = [best / n_cases]
    votes = one_hot[first].copy()

    while len(selected) < n_candidates:
        best_j, best_score = None, -1
        for j in range(n_candidates):
            if j in selected:
                continue
            score = _n_correct(votes + one_hot[j], y_index)
            if score > best_score:  # strict, so the first candidate wins ties
                best_j, best_score = j, score
        if best_score <= best:
            break
        selected.append(best_j)
        votes += one_hot[best_j]
        best = best_score
        scores.append(best / n_cases)

    return selected, scores
