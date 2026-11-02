# Lab 5 (Weeks 6 and 7): classifier evaluation and comparison

Read [Lab Sheet 5](../../Lab%20Sheet%205.md) for the exercises. Run these commands from the repository root after installing the packages in the [main README](../../README.md):

```bash
python code/lab5/single_problem.py
python code/lab5/bakeoff.py
```

These are **student starters**. The data loading, splits and file handling run, but the functions marked `NotImplementedError` are yours to complete. Each file has a `check_*` function with hand-worked answers. Save your results, plots and conclusions in `code/lab5/results`, which `.gitignore` excludes.

`plot_pairwise_scatter` needs `seaborn`, which the main install command includes. If you installed the packages before Lab 5, run `python -m pip install seaborn`.

## Task 1: one problem (`single_problem.py`)

The loaders return `X`, `y` and the class names. Breast cancer is already recoded so that malignant is the positive class 1. `single_split` returns the train and test row indices, which are also the case ids to save. `make_folds` gives the 5-fold `StratifiedKFold` to reuse for every classifier.

Replace the `"Yours"` stand-in in `make_classifiers` with a classifier you implemented, such as your Lab 3 `BasicEnsembleClassifier`. `bakeoff.py` imports the same function, so Task 2 uses the same three classifiers.

Implement `compute_metrics`, then call `check_metrics()`. `labelled_confusion_matrix` and `save_predictions` are provided. Use `class_column(model, label)` to find a class's probability column instead of assuming the order. Then implement `cross_val_predictions` for 1.2 and `tune` for 1.3. `PARAM_GRIDS` holds the Random Forest grid from the sheet; add one for your second classifier.

## Task 2: many problems (`bakeoff.py`)

The 35 UCI datasets are in `data/lab5/<name>/<name>.arff`, and `load_dataset(name)` reads one. Implement `run_one_dataset`. `run_experiments` appends each dataset's results to a CSV as it finishes and skips datasets already in the file, so you can stop and restart. Check your code on five datasets with a separate file before the full run. `results_table` turns the results into one row per dataset and one column per classifier.

Implement `average_ranks` and `win_draw_loss`, then call `check_functions()`. Rotation Forest is the slowest of the three; on the larger datasets each fit takes several seconds.
