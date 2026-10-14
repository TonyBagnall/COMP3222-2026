# Lab 4 (Week 5): time series classification

Read [Lab Sheet 4](../../Lab%20Sheet%204.pdf) for the exercises. Run these commands from the repository root after installing the packages in the [main README](../../README.md):

```bash
python code/lab4/lab4_task1.py
python code/lab4/lab4_task2.py
python code/lab4/lab4_task3.py
```

These are **student starters**. The data loading runs, but the functions marked `NotImplementedError` are yours to complete. Each file has a `check_*` function that compares your code with the worked examples in the Week 4 lectures.

The worked solutions are `lab4_task1_solution.py`, `lab4_task2_solution.py` and `lab4_task3_solution.py` in this directory. They show one way to complete the tasks after you have attempted them.

The GunPoint and BasicMotions data are bundled with `aeon`, so `load_gunpoint` and `load_basic_motions` do not need a download. Both return a 3D array of shape `(n_cases, n_channels, n_timepoints)` and a 1D array of string labels.

## Task 1: distances and nearest neighbours

Implement `euclidean_distance` and `dtw_distance` in `lab4_task1.py`; you can reuse your Lab 1 code. Then call `check_distances()`, which compares them with `aeon.distances` on the example from L4.1. Use `@njit(cache=True)` from Numba. Your 1-NN classifier will call DTW 7,500 times on GunPoint, which is slow in pure Python.

## Task 2: summary and interval features

`slope` is provided. Implement `summary_features`, `SummaryTransformer` and `RandomIntervalTransformer` in `lab4_task2.py`, and call `check_functions()` for the toy series on L4.2 slides 8 and 18. Fit the interval positions in `fit` and use the same positions for every case in `transform`.

## Task 3: shapelet distance and convolution from scratch

Implement `z_normalise`, `distance_profile`, `shapelet_distance`, `apply_kernel` and `max_and_ppv` in `lab4_task3.py`. Then call `check_functions()` to compare with L4.2 slides 24 and 30. Plot the distance profile and activation map of the same pattern on one case from each class, as described on the sheet. `RDSTClassifier` can take a minute or two to fit.

The optional extension uses `information_gains` in the same file; `entropy` is provided.

You do not need `bakeoff.py`, `classifier_compare.py` or `single_problem.py` for this lab.
