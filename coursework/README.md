# Coursework

Files for the COMP3222 coursework. The briefs explain what to submit.

- [Coursework 3: Tune or Combine](#coursework-3-tune-or-combine)
- [Ensemble Selection](#ensemble-selection)

All coursework uses the same [`requirements.txt`](requirements.txt) and the same data folder, [`data/ucr`](data/ucr).

## Setting up

Create a project folder for your submission, then copy `requirements.txt`, the setup files for your coursework and the `data` folder into it. From that folder, create a virtual environment with Python 3.11, 3.12 or 3.13 and install the packages.

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The required versions are pinned in `requirements.txt`. In particular, use `aeon==1.6.0`: distance implementations change between aeon releases, and we will compare your accuracies with results from this version. The reference results were produced with aeon 1.6.0, numba 0.63.1 and numpy 2.3.5.

The setup modules read the datasets from `data/ucr` next to the module. If you would rather not copy the data, set the environment variable `COMP3222_UCR_DIR` to the `coursework/data/ucr` folder of your clone of this repository instead. If a dataset is missing from that folder, it is downloaded from the UCR archive and saved there, which needs an internet connection. Use the copies in this repository where you can: they are the data your submission will be marked with.

Do not change the setup files or the data, and do not submit them.

## Coursework 3: Tune or Combine

| File | Purpose |
| --- | --- |
| [`coursework3_setup.py`](coursework3_setup.py) | Datasets, resamples, distances, parameter grids and loading |
| [`coursework3_greedy.py`](coursework3_greedy.py) | Reference greedy ensemble selection on precomputed LOOCV predictions |
| [`tests/test_coursework3_greedy.py`](tests/test_coursework3_greedy.py) | Tests for the greedy selector |

Lay out your folder like this:

```
code/
    coursework3_setup.py
    coursework3_greedy.py
    (your scripts)
data/
    ucr/
        ArrowHead/
        ...
requirements.txt
```

### What the setup module provides

| Name | Contents |
| --- | --- |
| `DATASETS` | The 20 UCR problems to use, all of them |
| `N_RESAMPLES` | The number of resamples, 10. Resample 0 is the default split |
| `DTW_WINDOWS` | The DTW window grid for Part A, ascending and ending with the full window |
| `GRIDS` | For each distance, its parameter grid as a list of keyword dictionaries for `aeon.distances` |
| `DISTANCES` | The distance names in tie-breaking order |
| `load_resample(name, resample)` | One resample as `X_train, y_train, X_test, y_test` |

`GRIDS` and `DISTANCES` are in tie-breaking order: if two settings give the same LOOCV accuracy, use the one listed first.

### Loading a resample

```python
from coursework3_setup import load_resample

X_train, y_train, X_test, y_test = load_resample("GunPoint", 3)
print(X_train.shape)  # (50, 1, 150): 50 cases, 1 channel, length 150
```

The X arrays have shape `(n_cases, 1, series_length)`, the shape aeon uses for a collection of univariate series. Resample 0 is the default train/test split. Resample `i > 0` comes from `aeon.benchmarking.resampling.stratified_resample_data` with `random_state=i`, applied to the default split.

`load_resample` returns the data as stored in the archive. Most of these datasets are already z-normalised, but BME, Chinatown, SmoothSubspace and UMD are not. Normalising is your choice. If you do normalise, use aeon's function on both `X_train` and `X_test`, apply no other preprocessing, and state in your README that you normalised, because it changes the accuracies we compare yours with:

```python
from aeon.utils.numba.general import z_normalise_series_3d

X_train = z_normalise_series_3d(X_train)
X_test = z_normalise_series_3d(X_test)
```

Use this function rather than writing your own: a numpy version differs in the last decimal place, which is enough to change some nearest neighbours.

### Distance matrices and LOOCV

Compute each distance matrix once and reuse it. For one distance and one parameter setting:

```python
import numpy as np
from aeon.distances import pairwise_distance

from coursework3_setup import GRIDS, load_resample

X_train, y_train, X_test, y_test = load_resample("GunPoint", 0)
params = GRIDS["dtw"][6]  # {"window": 0.1}

# Train-train and test-train matrices. n_jobs spreads the work over cores.
D_train = pairwise_distance(X_train, method="dtw", n_jobs=4, **params)
D_test = pairwise_distance(X_test, X_train, method="dtw", n_jobs=4, **params)

# LOOCV 1-NN: a case must never be matched to itself, so ignore the diagonal.
D_loocv = D_train.copy()
np.fill_diagonal(D_loocv, np.inf)
loocv_pred = y_train[np.argmin(D_loocv, axis=1)]

# Test 1-NN. np.argmin returns the first minimum, so equally close training
# cases go to the one with the lower index, as the brief requires.
test_pred = y_train[np.argmin(D_test, axis=1)]

print("LOOCV accuracy", np.mean(loocv_pred == y_train))
print("Test accuracy ", np.mean(test_pred == y_test))
```

Compute distances with `pairwise_distance`, or use `KNeighborsTimeSeriesClassifier`, which gives the same results. Do not build matrices from single-pair calls such as `ddtw_distance` or `distance`: in aeon 1.6.0, single-pair DDTW gives different values from `pairwise_distance` for some windows, and your results would not match ours.

The first call to each distance is slower because numba compiles it. Leave that time out of your timings, for example by calling every distance once on a tiny array before you start the clock.

Computing every matrix for all datasets and resamples takes a few hours of processor time, so start early and save the matrices to disk (for example with `np.save`) rather than computing them again.

### Greedy selection

```python
import numpy as np
from coursework3_greedy import greedy_select

# loocv_preds: one row of LOOCV predictions per candidate
selected, scores = greedy_select(loocv_preds, y_train, np.unique(y_train))
```

`selected` gives the candidate indices in the order they were added and `scores[i]` is the LOOCV accuracy of the vote of the first `i + 1` of them. The rules are those of the Ensemble Selection coursework. To run its tests, run `python -m pytest tests` in the `coursework` folder of your clone of this repository.

## Ensemble Selection

| File | Purpose |
| --- | --- |
| [`part3_setup.py`](part3_setup.py) | Datasets, classifier pool and settings for Part 3 |

Lay out your folder like this, with `part3_setup.py` and `data/` next to `solution/`:

```
data/
    ucr/
        Computers/
        ...
part3_setup.py
requirements.txt
solution/
tests/
```

`part3_setup.load_dataset(name)` reads the Part 3 datasets from `data/ucr` as described in [Setting up](#setting-up).
