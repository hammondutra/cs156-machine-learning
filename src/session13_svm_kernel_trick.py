"""CS156 Session 13 — SVCs, kernel trick, optional Yosemite SVR.
Exported from the matching Session 13 notebook; # %% boundaries work in VS Code.
Set CS156_RUN_YOSEMITE=1 to download NOAA data and run the expensive extensions.
"""

# %% [markdown]
# # CS156 — Session 13: Support Vector Classifiers and the Kernel Trick
# 
# Source: supplied Session 13 PCW PDF (pages 1–6), plus the five code excerpts provided separately for this repository update. Reading: Murphy (2012), Machine Learning: A Probabilistic Perspective, Section 14.4.
# 
# This session report includes the two Question 1 SVC functions that were missing from the original PCW code cells, the synthetic kernel comparisons, the optional financial-data experiment, the Yosemite month classifier, and a supplementary SVR extension.
# 
# **Reproducibility:** the synthetic examples run offline. The transaction experiment needs a CSV that was not provided. The Yosemite exercise downloads NOAA observations only after setting the environment variable CS156_RUN_YOSEMITE=1. The long-running data-dependent exercises are disabled by default. Numerical results transcribed from the PCW are labeled accordingly. Sessions 10–12 are intentionally not covered.
# 

# %% [markdown]
# ## Question 1 — Basic prep and SVC mathematics
# 
# 1. The nullspace of a design matrix X is N(X) = {v : Xv=0}. Adding a nullspace vector to the weights does not change training predictions. With L2 regularization, such a term needlessly raises the penalty.
# 
# 2. The representer theorem (in standard norm-regularized settings) lets us choose an optimum in the span of the training examples: w* = X.T @ alpha = sum_i alpha_i x_i. This does **not** mean every optimum of every possible unregularized objective must be in that span.
# 
# 3. Prompts used: “Write a Python function using scikit-learn that trains and returns a linear Support Vector Classifier (SVC). Explain the parameters and how the classifier works.” / “Modify the function to use a radial basis function (RBF) kernel instead of a linear kernel. Explain how gamma and C influence the model's decision boundary.”
# 
# 4. Linear prediction is sign(w.T @ x + b). A soft-margin SVC minimizes (1/2)||w||² + C Σ ξ_i subject to y_i(w.T x_i + b) ≥ 1 − ξ_i and ξ_i ≥ 0. The kernelized predictor is sign(Σ alpha_i y_i K(x_i,x) + b); the RBF kernel is exp(−gamma ||x−z||²). C controls the margin/violation tradeoff; gamma controls the locality of the RBF kernel. A standard SVM kernel must give a positive-semidefinite Gram matrix.
# 

# %%
# Code Cell 01 — Original Q1 linear and RBF SVC helpers (not in PCW code cells)
from sklearn.svm import SVC

def linear_svc(X, y, C=1.0):
    model = SVC(kernel="linear", C=C)
    model.fit(X, y)
    return model

def rbf_svc(X, y, C=1.0, gamma="scale"):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    model.fit(X, y)
    return model

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
linear_model = linear_svc(X, y)
rbf_model = rbf_svc(X, y)
print("Linear SVC training accuracy:", accuracy_score(y, linear_model.predict(X)))
print("RBF SVC training accuracy:", accuracy_score(y, rbf_model.predict(X)))


# %% [markdown]
# These first two scores are **training** accuracies: no hold-out was used in Code Cell 01.
# 
# ## Question 2 — Kernel feature maps
# 
# - Linear: K(x,z) = x.T @ z; feature map Phi(x)=x.
# - Homogeneous quadratic: K(x,z)=(x.T @ z)². For x=(x1,x2), Phi(x)=(x1², sqrt(2) x1 x2, x2²). In sklearn, this precise map corresponds to polynomial degree=2, gamma=1, coef0=0. An inhomogeneous kernel (x.T @ z+1)² has additional constant and linear coordinates.
# - RBF: K(x,z)=exp(−gamma ||x−z||²), with an infinite-dimensional implicit feature map.
# 
# The kernel trick substitutes K(x,z)=Phi(x).T @ Phi(z) whenever the learning algorithm can use dot products, avoiding explicit high-dimensional features.
# 

# %%
# Code Cell 02 — Basic kernel SVM with a held-out test set
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Linear SVM accuracy:", accuracy_score(y_test, predictions))


# %%
# Code Cell 03 — Compare linear, quadratic and RBF kernels
kernels = {
    "Linear": {"kernel": "linear"},
    "Quadratic": {"kernel": "poly", "degree": 2, "gamma": 1.0, "coef0": 0.0},
    "RBF": {"kernel": "rbf", "gamma": "scale"}
}

def train_kernel_svm(X_train, y_train, kernel_name, C=1.0):
    model = make_pipeline(
        StandardScaler(),
        SVC(C=C, **kernels[kernel_name])
    )
    model.fit(X_train, y_train)
    return model

for name in kernels:
    model = train_kernel_svm(X_train, y_train, name)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name}: Accuracy = {accuracy:.4f}")


# %% [markdown]
# ### Results recorded in PCW (page 3)
# 
# The held-out make_moons accuracies were **linear 0.8533**, **quadratic 0.4800**, **RBF 0.9067**. These are recorded source outputs, not newly executed results. The homogeneous quadratic kernel performs particularly poorly on this split; nonlinear kernels are not guaranteed to outperform linear kernels on every dataset.
# 
# ### Financial transactions
# 
# The PCW guide refers to a financial transaction dataset, but no transaction CSV was supplied. Code Cell 04 preserves the scikit-learn experiment, with an optional-file guard so the notebook runs without falsely claiming to have tested the missing data. By default it looks for transactions.csv with target column Fraud.
# 

# %%
# Code Cell 04 — Compare kernels on financial transactions (external CSV required)
import os
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

CSV_PATH = Path(os.getenv("CS156_TRANSACTIONS_CSV", "transactions.csv"))
TARGET = os.getenv("CS156_FINANCIAL_TARGET", "Fraud")

def compare_financial_kernels(csv_path=CSV_PATH, target=TARGET):
    df = pd.read_csv(csv_path).dropna(subset=[target])
    X_fin = df.drop(columns=[target])
    y_fin = df[target]
    X_fin_train, X_fin_test, y_fin_train, y_fin_test = train_test_split(
        X_fin, y_fin, test_size=0.25, random_state=42, stratify=y_fin
    )
    numeric_cols = X_fin.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_fin.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler()
        ), numeric_cols),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        ), categorical_cols)
    ])
    results = []
    for name, params in kernels.items():
        model = Pipeline([
            ("preprocess", preprocessor),
            ("svm", SVC(C=1.0, **params))
        ])
        model.fit(X_fin_train, y_fin_train)
        predictions = model.predict(X_fin_test)
        results.append({
            "Kernel": name,
            "Accuracy": accuracy_score(y_fin_test, predictions),
            "Balanced Accuracy": balanced_accuracy_score(y_fin_test, predictions),
            "Macro F1": f1_score(
                y_fin_test, predictions, average="macro", zero_division=0
            )
        })
    results_df = pd.DataFrame(results).sort_values(
        "Accuracy", ascending=False
    )
    print(results_df.to_string(index=False))
    best_accuracy = results_df["Accuracy"].max()
    winners = results_df.loc[
        results_df["Accuracy"] == best_accuracy, "Kernel"
    ].tolist()
    print("\nHighest test accuracy:", best_accuracy)
    print("Best kernel(s):", ", ".join(winners))
    return results_df

if CSV_PATH.is_file():
    financial_results = compare_financial_kernels()
else:
    print(f"Financial experiment skipped: {CSV_PATH} not found.")


# %% [markdown]
# ## Question 3 — Yosemite extension
# 
# **Month classification:** predict the month (1–12) from *raw temperature alone*, not the date, using 2011–2015 Yosemite observations for training and 2016 for testing. We compare linear, homogeneous quadratic, and RBF SVCs, tuning C and RBF gamma by threefold CV on the training sample (4,000 points), then evaluate on a fixed 2016 test sample (3,000 points).
# 
# Temperature is an ambiguous month indicator because observations from different months overlap. This is a time-based hold-out, not a random year-mixing split. The NOAA download is opt-in. Set CS156_RUN_YOSEMITE=1 in the environment to run Code Cell 05 and the supplementary SVR extension.
# 

# %%
# Code Cell 05 — Yosemite month classification
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

RUN_YOSEMITE = os.getenv("CS156_RUN_YOSEMITE", "0") == "1"

def load_yosemite_observations():
    """Load YYYYMMDD, HHmm, air temperature from NOAA USCRN 2011–2016."""
    if "data" in globals() and "valid" in globals():
        obs = np.asarray(data)[np.asarray(valid)]
    else:
        records = []
        for year in range(2011, 2017):
            url = (
                "https://www.ncei.noaa.gov/pub/data/uscrn/"
                f"products/subhourly01/{year}/"
                f"CRNS0101-05-{year}-CA_Yosemite_Village_12_W.txt"
            )
            records.append(
                pd.read_csv(
                    url, sep=r"\s+", header=None, usecols=[1, 2, 8]
                ).to_numpy()
            )
        obs = np.vstack(records)
    obs = np.asarray(obs, dtype=float)
    return obs[np.isfinite(obs).all(axis=1) & (obs[:, 2] > -1000)]

def classify_yosemite_months(obs):
    dates = obs[:, 0].astype(int)
    years = dates // 10000
    months = (dates // 100) % 100
    temps = obs[:, 2]
    mask = (months >= 1) & (months <= 12)
    dates, years, months, temps = (
        arr[mask] for arr in (dates, years, months, temps)
    )

    train_pool = np.where((years >= 2011) & (years <= 2015))[0]
    test_pool = np.where(years == 2016)[0]
    train_idx, _ = train_test_split(
        train_pool, train_size=4000,
        stratify=months[train_pool], random_state=42
    )
    test_idx, _ = train_test_split(
        test_pool, train_size=3000,
        stratify=months[test_pool], random_state=42
    )
    X_train_month = temps[train_idx].reshape(-1, 1)
    X_test_month = temps[test_idx].reshape(-1, 1)
    y_train_month = months[train_idx]
    y_test_month = months[test_idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_month, y_train_month)
    print("Baseline accuracy:",
          round(dummy.score(X_test_month, y_test_month), 4))

    settings = {
        "Linear": {"kernel": "linear", "C": [0.1, 1, 10]},
        "Quadratic": {
            "kernel": "poly", "degree": 2,
            "gamma": 1.0, "coef0": 0.0, "C": [0.1, 1, 10]
        },
        "RBF": {
            "kernel": "rbf", "C": [0.1, 1, 10],
            "gamma": [0.1, 1.0]
        }
    }
    results = []
    for name, params in settings.items():
        kernel = params["kernel"]
        fixed = {k: params[k] for k in
                 ["degree", "gamma", "coef0"]
                 if k in params and name == "Quadratic"}
        model = make_pipeline(
            StandardScaler(), SVC(kernel=kernel, cache_size=50, **fixed)
        )
        grid = {"svc__C": params["C"]}
        if name == "RBF":
            grid["svc__gamma"] = params["gamma"]
        search = GridSearchCV(
            model, grid, cv=3, scoring="accuracy", n_jobs=1
        )
        search.fit(X_train_month, y_train_month)
        accuracy = accuracy_score(
            y_test_month, search.predict(X_test_month)
        )
        results.append({
            "Kernel": name,
            "CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy
        })
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    best = results_df.loc[results_df["CV Accuracy"].idxmax()]
    print("\nBest kernel by cross-validation:", best["Kernel"])
    print("Its 2016 test accuracy:", round(best["Test Accuracy"], 4))
    return {
        "results": results_df, "dates": dates,
        "temps": temps, "train_idx": train_idx,
        "test_idx": test_idx, "good_mask": mask
    }

yosemite_result = None
if RUN_YOSEMITE:
    try:
        yosemite_obs = load_yosemite_observations()
        yosemite_result = classify_yosemite_months(yosemite_obs)
    except Exception as exc:
        print("Yosemite experiment could not complete:", repr(exc))
else:
    print("Yosemite experiment skipped; set CS156_RUN_YOSEMITE=1.")


# %% [markdown]
# ### Yosemite outcomes copied from source PCW (pages 5–6)
# 
# | Method | 3-fold CV accuracy | 2016 test accuracy |
# | --- | ---: | ---: |
# | Most-common-month baseline | n/a | 0.0850 |
# | Linear SVC | 0.193251 | **0.174667** |
# | Quadratic SVC | 0.111999 | 0.134000 |
# | RBF SVC | **0.202252** | 0.157333 |
# 
# **Interpretation:** RBF was selected by CV but the linear kernel produced the highest held-out test accuracy among the three. Selecting a kernel using test-set scores would leak test information. Month cannot be determined reliably from raw temperature alone.
# 
# ### Supplement: SVR against linear and RBF-feature models
# 
# The source PDF has a text response explaining SVR and cites a historical linear-regression R² about **0.0172**, but its bonus code cell is empty. The following cell implements the planned regression comparison; **there are no recorded SVR scores in the source document**. Historical 0.0172 is from a different earlier evaluation, not the same 2016 split. Here all regressors use the same hold-out and test R² = 1 − SSR/SST.
# 

# %%
# Supplementary cell — SVR and linear-parameter models (not in source PDF)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel

def compare_yosemite_regressors(obs, result):
    mask = result["good_mask"]
    dates = obs[:, 0].astype(int)[mask]
    hhmm = obs[:, 1].astype(int)[mask]
    minutes = (hhmm // 100) * 60 + hhmm % 100
    temps = result["temps"]
    train_idx, test_idx = result["train_idx"], result["test_idx"]
    y_reg_train, y_reg_test = temps[train_idx], temps[test_idx]

    X_time_train = minutes[train_idx].reshape(-1, 1)
    X_time_test = minutes[test_idx].reshape(-1, 1)
    linear = LinearRegression().fit(X_time_train, y_reg_train)
    svr_time = make_pipeline(
        StandardScaler(), SVR(kernel="rbf", C=10, epsilon=1.0)
    ).fit(X_time_train, y_reg_train)

    def cyclic_features(indices):
        d = pd.to_datetime(
            dates[indices].astype(str), format="%Y%m%d"
        )
        doy = d.dayofyear.to_numpy()
        days_in_year = np.where(d.is_leap_year, 366, 365)
        daily = 2 * np.pi * minutes[indices] / 1440
        annual = 2 * np.pi * (doy - 1) / days_in_year
        return np.column_stack([
            np.sin(daily), np.cos(daily),
            np.sin(annual), np.cos(annual)
        ])

    Z_train, Z_test = cyclic_features(train_idx), cyclic_features(test_idx)
    centers = Z_train[np.linspace(0, len(Z_train) - 1, 60, dtype=int)]
    Phi_train = rbf_kernel(Z_train, centers, gamma=1.0)
    Phi_test = rbf_kernel(Z_test, centers, gamma=1.0)

    ridge = Ridge(alpha=1.0).fit(Phi_train, y_reg_train)
    svr_seasonal = make_pipeline(
        StandardScaler(), SVR(kernel="rbf", C=10, epsilon=1.0)
    ).fit(Z_train, y_reg_train)

    scores = {
        "Linear (time only)": r2_score(
            y_reg_test, linear.predict(X_time_test)
        ),
        "SVR (time only)": r2_score(
            y_reg_test, svr_time.predict(X_time_test)
        ),
        "RBF-feature Ridge (seasonal)": r2_score(
            y_reg_test, ridge.predict(Phi_test)
        ),
        "SVR (seasonal)": r2_score(
            y_reg_test, svr_seasonal.predict(Z_test)
        )
    }
    for name, score in scores.items():
        print(f"{name}: R² = {score:.4f}")
    print("Best R² model:", max(scores, key=scores.get))
    return scores

if yosemite_result is not None:
    svr_results = compare_yosemite_regressors(
        yosemite_obs, yosemite_result
    )
else:
    print("SVR extension skipped; run Yosemite first.")


# %% [markdown]
# ## Takeaways and next steps
# 
# - A norm-regularized SVM weight vector can be represented in the training-point span; kernels make the corresponding similarity computations possible implicitly.
# - On synthetic moons, the recorded RBF test accuracy was 90.67%.
# - On Yosemite, month prediction was difficult (best test accuracy 17.47%), and CV and held-out ranking disagreed.
# - Finance needs the actual transactions CSV and target definition; no finance results have been invented.
# - SVR is included as a newly supplied, optional exercise, rather than as an experiment claimed completed in the original PCW.
# - Keep preprocessing inside each training pipeline, select hyperparameters using training CV, and report final unseen-test metrics without using them to select hyperparameters.
# 
# Sessions 10–12 will be incorporated later.
# 
