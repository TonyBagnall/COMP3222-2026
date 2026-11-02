# COMP3222 Machine Learning Technologies — 2026 labs

This repository contains lab sheets, starter code, notebooks, and datasets for COMP3222.

| Start here | Material |
| --- | --- |
| [Lab 1 guide](code/lab1/README.md) | [Lab Sheet 1](Lab%20Sheet%201.pdf), [starter example](code/lab1/lab_1.py), [notebook](notebooks/lab1.ipynb), and solutions in `code/lab1` |
| [Lab 2 guide](code/lab2/README.md) | [Lab Sheet 2](Lab%20Sheet%202.pdf), starter code in `code/lab2`, and data in `data/lab2` |
| [Lab 3 guide](code/lab3/README.md) | [Lab Sheet 3](Lab%20Sheet%203.pdf), starter code in `code/lab3`, and data in `data/lab3` |
| [Lab 4 guide](code/lab4/README.md) | [Lab Sheet 4](Lab%20Sheet%204.pdf), starter code in `code/lab4`; data is bundled with `aeon` |
| [Lab 5 guide](code/lab5/README.md) | [Lab Sheet 5](Lab%20Sheet%205.pdf), starter code in `code/lab5`, and data in `data/lab5` |
| [Coursework guide](coursework/README.md) | Setup, package versions and `part3_setup.py` for the Ensemble Selection coursework |
| [Jupyter introduction](notebooks/labW1_Intro_to_Jupyter.ipynb) | Optional NumPy and notebook practice |

Each lab's code is in `code/labN`, with any data files in `data/labN`. The lab sheet PDFs are at the repository root.

## Getting started

Run commands from the repository root unless a lab guide says otherwise. Create and activate a virtual environment, then install the packages used in the labs:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn numba matplotlib seaborn jupyter aeon
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn numba matplotlib seaborn jupyter aeon
```

Use the guides above for each lab's tasks and run commands.
