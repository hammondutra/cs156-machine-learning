"""Multivariate linear-regression utilities from CS156 Session 3."""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


TITANIC_URL = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/titanic.csv"
)


def load_titanic_fare_data(url: str = TITANIC_URL):
    """Load Titanic data and return the three PCW predictors plus Fare."""
    titanic = pd.read_csv(url)
    df = titanic[["Fare", "Pclass", "Age", "SibSp"]].dropna().copy()
    X = df[["Pclass", "Age", "SibSp"]]
    y = df["Fare"]
    return df, X, y


def fit_fare_model(X, y) -> LinearRegression:
    """Fit sklearn OLS with an intercept."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def ols_closed_form(X, y):
    """Return least-squares coefficients using np.linalg.lstsq."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def sum_squared_residuals(y, y_pred) -> float:
    """Return r^T r."""
    residuals = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(residuals.T @ residuals)


def three_passenger_example():
    """Return the three observations used in the Session 3 PCW."""
    X = np.array(
        [
            [3.0, 22.0, 1.0],
            [1.0, 38.0, 1.0],
            [3.0, 26.0, 0.0],
        ]
    )
    y = np.array([7.25, 71.2833, 7.925])
    beta = np.linalg.solve(X, y)
    return X, y, beta
