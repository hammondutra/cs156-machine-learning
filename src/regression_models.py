"""Linear and logistic regression utilities from CS156 Session 2."""

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression


def load_iris_dataframe() -> pd.DataFrame:
    """Load Iris and add species names."""
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))
    return df


def fit_linear_regression(
    df: pd.DataFrame,
    x_col: str = "petal length (cm)",
    y_col: str = "sepal length (cm)",
) -> LinearRegression:
    """Fit sklearn linear regression using one predictor."""
    model = LinearRegression()
    model.fit(df[[x_col]], df[y_col])
    return model


def fit_logistic_regression(
    df: pd.DataFrame,
    x_col: str = "petal length (cm)",
    positive_species: str = "virginica",
) -> LogisticRegression:
    """Fit virginica-vs-rest logistic regression using one predictor."""
    y = (df["species_name"] == positive_species).astype(int)
    model = LogisticRegression()
    model.fit(df[[x_col]], y)
    return model


def sensitivity_score(y_true, y_pred) -> float:
    """Return TP / (TP + FN)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn)


def linear_regression_from_scratch(x, y):
    """Closed-form one-dimensional OLS with an intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum(
        (x - x_mean) ** 2
    )
    intercept = y_mean - slope * x_mean

    predictions = intercept + slope * x
    mse = np.mean((y - predictions) ** 2)

    return intercept, slope, predictions, mse


def logistic_regression_from_scratch(
    x,
    y,
    learning_rate: float = 0.05,
    iterations: int = 20_000,
):
    """One-feature binary logistic regression trained with gradient descent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    intercept = 0.0
    slope = 0.0

    for _ in range(iterations):
        z = np.clip(intercept + slope * x, -500, 500)
        probabilities = 1 / (1 + np.exp(-z))
        error = probabilities - y

        intercept -= learning_rate * np.mean(error)
        slope -= learning_rate * np.mean(error * x)

    z = np.clip(intercept + slope * x, -500, 500)
    probabilities = 1 / (1 + np.exp(-z))
    predictions = (probabilities >= 0.5).astype(int)

    return intercept, slope, probabilities, predictions
