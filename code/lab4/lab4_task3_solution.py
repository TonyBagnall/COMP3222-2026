"""Lab 4, Task 3 solution: shapelet distance and convolution from scratch."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.datasets import load_gunpoint

from lab4_task3 import entropy


def z_normalise(x):
    std = x.std()
    return np.zeros(len(x)) if std == 0 else (x - x.mean()) / std


def distance_profile(shapelet, series):
    shapelet = z_normalise(shapelet)
    L = len(shapelet)
    return np.array([
        np.sqrt(((z_normalise(series[i:i + L]) - shapelet) ** 2).sum())
        for i in range(len(series) - L + 1)
    ])


def shapelet_distance(shapelet, series):
    return distance_profile(shapelet, series).min()


def apply_kernel(series, kernel):
    L = len(kernel)
    return np.array([np.dot(series[i:i + L], kernel) for i in range(len(series) - L + 1)])


def max_and_ppv(series, kernel):
    activation = apply_kernel(series, kernel)
    return activation.max(), np.mean(activation > 0)


def information_gains(distances, y):
    order = np.argsort(distances)
    distances, y = np.asarray(distances)[order], np.asarray(y)[order]
    n = len(y)
    return [
        ((distances[k - 1] + distances[k]) / 2,
         entropy(y) - k / n * entropy(y[:k]) - (n - k) / n * entropy(y[k:]))
        for k in range(1, n)
    ]


if __name__ == "__main__":
    import lab4_task3

    for name in ["z_normalise", "distance_profile", "shapelet_distance",
                 "apply_kernel", "max_and_ppv"]:
        setattr(lab4_task3, name, globals()[name])
    lab4_task3.check_functions()

    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    shapelet = X_train[5, 0, 100:140]
    kernel = z_normalise(shapelet)

    fig, axes = plt.subplots(3, 1, sharex=True)
    for label in np.unique(y_train):
        series = X_train[y_train == label][0, 0]
        axes[0].plot(series, label=f"class {label}")
        axes[1].plot(distance_profile(shapelet, series))
        axes[2].plot(apply_kernel(series, kernel))
    axes[0].set_title("Series")
    axes[1].set_title("Distance profile")
    axes[2].set_title("Activation map")
    axes[0].legend()
    plt.tight_layout()
    plt.show()

    s_dist = np.array([shapelet_distance(shapelet, x[0]) for x in X_train])
    max_activation = np.array([apply_kernel(x[0], kernel).max() for x in X_train])
    for label in np.unique(y_train):
        mask = y_train == label
        plt.scatter(s_dist[mask], max_activation[mask], label=f"class {label}")
    plt.xlabel("sDist")
    plt.ylabel("Max activation")
    plt.legend()
    plt.show()

    for clf in [RocketClassifier(n_kernels=2000, random_state=0),
                RDSTClassifier(random_state=0)]:
        clf.fit(X_train, y_train)
        print(f"{type(clf).__name__}: {accuracy_score(y_test, clf.predict(X_test)):.3f}")

    # Optional extension: best information gain of each feature.
    for name, values in [("sDist", s_dist), ("max activation", max_activation)]:
        print(f"Best gain, {name}: {max(g for _, g in information_gains(values, y_train)):.3f}")
