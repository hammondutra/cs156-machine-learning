"""Reusable utilities from CS156 Session 1: introductory data representations."""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml, load_iris


def load_iris_dataframe() -> pd.DataFrame:
    """Return the Iris dataset with a human-readable species column."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df


def filter_iris_by_sepal_length(
    df: pd.DataFrame,
    maximum: float = 5.0,
) -> pd.DataFrame:
    """Keep Iris observations whose sepal length is at most `maximum` cm."""
    return df[df["sepal length (cm)"] <= maximum].copy()


def load_mnist_3_and_8() -> pd.DataFrame:
    """Download MNIST from OpenML and keep only handwritten 3s and 8s."""
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=True,
        return_X_y=True,
    )
    df = X.copy()
    df["digit"] = y.astype(int)
    return df[df["digit"].isin([3, 8])].copy()


def reshape_digit(row: pd.Series, label_col: str = "digit"):
    """Convert one flattened MNIST row back into a 28 x 28 array."""
    return row.drop(label_col).to_numpy(dtype=float).reshape(28, 28)


def plot_digit(row: pd.Series, label_col: str = "digit") -> None:
    """Display one MNIST digit."""
    image = reshape_digit(row, label_col=label_col)
    plt.imshow(image, cmap="gray")
    plt.title(f"Digit: {int(row[label_col])}")
    plt.axis("off")
    plt.show()
