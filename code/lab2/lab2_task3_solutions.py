"""Lab 2, Task 3: inspect split choices and tree regularisation.

The printed accuracies are on the training data. They show how closely each
tree fits these rows; choosing a model needs evaluation on unseen data.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from lab2_task2_solutions import load_playgolf, make_tree_pipeline
from lab2_task3 import load_data


def compare_playgolf_settings():
    """Task 3A: change split criterion and splitter on the small dataset."""
    X, y = load_playgolf()
    print("PlayGolf: criterion and splitter")
    for criterion, splitter in [
        ("gini", "best"),
        ("entropy", "best"),
        ("log_loss", "best"),
        ("gini", "random"),
    ]:
        pipeline = make_tree_pipeline(
            DecisionTreeClassifier(
                criterion=criterion, splitter=splitter, random_state=0
            )
        ).fit(X, y)
        fitted_tree = pipeline.named_steps["tree"]
        print(
            f"  {criterion:8s} {splitter:6s} "
            f"depth={fitted_tree.get_depth():2d} "
            f"leaves={fitted_tree.get_n_leaves():2d} "
            f"training accuracy={pipeline.score(X, y):.3f}"
        )


def compare_italy_settings():
    """Task 3B: fit a full tree, then restrict or prune its growth."""
    X, y = load_data()  # 1,096 cases, 24 hourly values per case.
    print(f"\nItalyPowerDemand: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

    settings = [
        ("unrestricted", {}),
        ("max_depth=4", {"max_depth": 4}),
        ("min_samples_split=20", {"min_samples_split": 20}),
        ("min_samples_leaf=10", {"min_samples_leaf": 10}),
        ("max_leaf_nodes=10", {"max_leaf_nodes": 10}),
        ("min_impurity_decrease=0.01", {"min_impurity_decrease": 0.01}),
        ("ccp_alpha=0.10", {"ccp_alpha": 0.10}),
        ("ccp_alpha=0.02", {"ccp_alpha": 0.02}),
    ]
    fitted = {}
    for name, parameters in settings:
        classifier = DecisionTreeClassifier(random_state=0, **parameters).fit(X, y)
        fitted[name] = classifier
        print(
            f"  {name:28s} depth={classifier.get_depth():2d} "
            f"leaves={classifier.get_n_leaves():2d} "
            f"training accuracy={classifier.score(X, y):.3f}"
        )

    # Show only three levels of the full tree so its text stays readable.
    feature_names = [f"hour_{i}" for i in range(1, X.shape[1] + 1)]
    season_names = {1: "Oct-Mar", 2: "Apr-Sep"}
    class_names = [season_names[int(label)] for label in fitted["unrestricted"].classes_]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    tree.plot_tree(
        fitted["unrestricted"],
        max_depth=2,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        fontsize=9,
        ax=axes[0],
    )
    axes[0].set_title("Full tree (top three levels shown)")
    tree.plot_tree(
        fitted["ccp_alpha=0.02"],
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        fontsize=9,
        ax=axes[1],
    )
    axes[1].set_title("Pruned tree (ccp_alpha=0.02)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_playgolf_settings()
    compare_italy_settings()
    print("Use unseen data, rather than training accuracy, to choose settings.")
