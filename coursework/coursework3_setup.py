"""Shared settings for COMP3222 coursework 3: Tune or Combine.

Import these definitions rather than copying or changing them, so that every
submission uses the same data, resamples, distances and parameter grids::

    from coursework3_setup import (DATASETS, DISTANCES, DTW_WINDOWS, GRIDS,
                                   N_RESAMPLES, load_resample)

The datasets are included in the course repository under ``data/ucr``.
They are read from the directory named by the environment variable
COMP3222_UCR_DIR if it is set, and from ``data/ucr`` next to this file
otherwise. A dataset that is not found there is downloaded from the UCR
archive and saved in that directory, which needs an internet connection.
"""

import os
from pathlib import Path

import numpy as np
from aeon.benchmarking.resampling import stratified_resample_data
from aeon.datasets import load_classification

DATASETS = [
    "ArrowHead", "BME", "CBF", "Chinatown", "Coffee",
    "ECG200", "ECGFiveDays", "FaceFour", "GunPoint", "ItalyPowerDemand",
    "MoteStrain", "Plane", "SmoothSubspace", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "SyntheticControl", "ToeSegmentation1",
    "TwoLeadECG", "UMD", "Wine",
]

N_RESAMPLES = 10  # resample 0 is the default train/test split

# DTW and DDTW windows as a proportion of series length, in ascending order,
# ending with the full window (no constraint).
DTW_WINDOWS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0]

# Distances in tie-breaking order. Each grid is also in tie-breaking order:
# if two settings give the same LOOCV accuracy, use the one listed first.
# The keyword names are those used by aeon.distances.pairwise_distance.
GRIDS = {
    "euclidean": [{}],
    "dtw": [{"window": w} for w in DTW_WINDOWS],
    "ddtw": [{"window": w} for w in DTW_WINDOWS],
    "wdtw": [{"g": g} for g in [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
                                0.25, 0.5, 1.0]],
    "lcss": [{"window": w, "epsilon": e}
             for w in [0.1, 0.25] for e in [0.05, 0.1, 0.25, 0.5, 1.0]],
    "erp": [{"window": w, "g": g}
            for w in [0.1, 1.0] for g in [0.0, 0.25, 0.5, 0.75, 1.0]],
    "msm": [{"c": c} for c in [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
                               5.0, 10.0]],
    "twe": [{"nu": nu, "lmbda": lm}
            for lm in [0.0, 0.1] for nu in [1e-4, 1e-3, 1e-2, 0.1, 1.0]],
}

DISTANCES = list(GRIDS)

DATA_DIR = Path(
    os.environ.get("COMP3222_UCR_DIR", Path(__file__).parent / "data" / "ucr")
)


def _load_split(name, split):
    X, y = load_classification(name, split=split, extract_path=DATA_DIR)
    if isinstance(X, list) or X.ndim != 3 or X.shape[1] != 1:
        raise ValueError(f"{name} is not an equal-length univariate problem.")
    return X.astype(np.float64), np.asarray(y)


def load_resample(name, resample):
    """Load one train/test resample of a dataset.

    Resample 0 is the default split from the archive. For resample i > 0,
    the default train and test data are combined and split again with
    ``stratified_resample_data(..., random_state=i)``, which keeps the
    train and test sizes and the class distributions.

    The data are returned as stored in the archive, without normalisation.
    If you normalise, apply ``z_normalise_series_3d`` from
    ``aeon.utils.numba.general`` to both X arrays, as the brief requires.

    Parameters
    ----------
    name : str
        A name from DATASETS.
    resample : int
        The resample number, from 0 to N_RESAMPLES - 1.

    Returns
    -------
    X_train, y_train, X_test, y_test : numpy.ndarray
        X arrays have shape (n_cases, 1, series_length): one univariate
        series per case, the shape aeon uses for collections.
        y arrays hold the class labels as strings.
    """
    X_train, y_train = _load_split(name, "train")
    X_test, y_test = _load_split(name, "test")
    if resample > 0:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test, random_state=resample
        )
    return X_train, y_train, X_test, y_test

