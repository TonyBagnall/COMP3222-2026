"""Lab 2, Task 2: encode PlayGolf and compare two decision trees."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"


def load_playgolf():
    """Keep Outlook as text so the preprocessing step is visible."""
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns="PlayGolf")
    y = data["PlayGolf"].to_numpy()
    return X, y


def make_tree_pipeline(classifier):
    """One-hot encode Outlook and pass the three numeric columns through."""
    prepare = ColumnTransformer(
        [("outlook", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["Outlook"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("prepare", prepare), ("tree", classifier)])


def plot_fitted_tree(pipeline, ax, title):
    """plot_tree needs the fitted tree inside the pipeline."""
    feature_names = pipeline.named_steps["prepare"].get_feature_names_out()
    tree.plot_tree(
        pipeline.named_steps["tree"],
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        ax=ax,
    )
    ax.set_title(title)


if __name__ == "__main__":
    X, y = load_playgolf()

    # The raw string values in Outlook cannot be passed to this classifier.
    try:
        DecisionTreeClassifier(random_state=0).fit(X, y)
    except ValueError as error:
        print("Raw data cannot be fitted:", error)

    models = [
        ("DecisionTree", DecisionTreeClassifier(random_state=0)),
        ("ExtraTree", ExtraTreeClassifier(random_state=0)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (name, classifier) in zip(axes, models):
        pipeline = make_tree_pipeline(classifier).fit(X, y)
        X_encoded = pipeline.named_steps["prepare"].transform(X)
        print(f"{name}: X shape {X_encoded.shape}, training accuracy {pipeline.score(X, y):.3f}")
        plot_fitted_tree(pipeline, ax, name)

    plt.tight_layout()
    plt.show()
    print("Training accuracy does not tell us which tree will work better on new data.")
