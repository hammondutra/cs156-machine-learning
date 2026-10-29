"""CS156 Yosemite temperature experiment: Fourier sine/cosine bases, Ridge, and validation.
# %% cells support VS Code interactive execution. See companion ipynb for explanations.
"""

# %% [markdown]
# # CS156 — Yosemite Temperature: Fourier Basis Functions + Ridge
#
# This is a **Yosemite-based replacement for the synthetic Bollywood data in the *new experiment***. The completed Session 15 PCW is preserved separately in the repository.
#
# **Objective:** predict the hourly air temperature (°C) from sine/cosine basis functions, and then improve prediction using Ridge regularization and historical temperature lags.
#
# **Source:** [Prof. Watson's Yosemite data Gist](https://gist.github.com/Prof-Watson/dedfbf237d013900fa5c4b54c5fd939a), a whitespace-delimited weather file (not a CSV and not the Gist HTML page). It includes local date (field 4), local HHMM (field 5), and air temperature (field 9), counting from 1.
#
# **Metric:** This is regression, so classification "accuracy" does not apply. Minimize **RMSE** and **MAE** (°C) and maximize **R²**. Use 2012–2014 as training history, 2015 to select features and hyperparameters, and **2016 as untouched test data**.
#
# **Forecast interpretation:** The lagged variants use actual temperatures from 24, 48, 72 or 168 hours before the prediction timestamp. Therefore this is a rolling **24-hour-ahead forecasting** experiment with those historical measurements available, not an advance prediction of all of 2016 at once. Do not use the current temperature or any future measurement as a feature.
#
# Run all cells in order. The data download requires internet access; this environment could not independently execute the full download, so no new performance metrics are claimed here.
#

# %% [markdown]
# ## Code Cell 1 — Download and parse Yosemite readings
#
# The Gist URL you supplied is a web page. `pd.read_csv(gist_page_url)` will not correctly load the table. Use its **raw** text URL with whitespace separation and no header row.
#
# We aggregate 5-minute observations to **hourly mean temperatures**. That makes 24-hour lags exactly 24 hourly rows and reduces fitting time.
#

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RAW_URL = (
    "https://gist.githubusercontent.com/Prof-Watson/"
    "dedfbf237d013900fa5c4b54c5fd939a/raw/yosemite_data.txt"
)

# The source has no column names: local date=3, local HHMM=4, air temp=8 (zero-based).
chunks = []
for chunk in pd.read_csv(
    RAW_URL, sep=r"\s+", header=None, usecols=[3, 4, 8],
    chunksize=100_000
):
    chunk.columns = ["local_date", "local_hhmm", "temperature"]
    chunk["temperature"] = pd.to_numeric(chunk["temperature"], errors="coerce")
    dates = chunk["local_date"].astype(str).str.zfill(8)
    times = chunk["local_hhmm"].astype(str).str.zfill(4)
    chunk["timestamp"] = pd.to_datetime(
        dates + times, format="%Y%m%d%H%M", errors="coerce"
    )
    good = (
        chunk["timestamp"].notna()
        & chunk["temperature"].between(-60, 60)
    )
    chunks.append(chunk.loc[good, ["timestamp", "temperature"]])

data = (
    pd.concat(chunks, ignore_index=True)
    .groupby("timestamp", as_index=True)["temperature"].mean()
    .sort_index()
)
hourly = data.resample("1h").mean().to_frame("temperature")
print("Available years:", hourly.index.year.value_counts().sort_index().to_dict())
print("Rows:", len(hourly), "Valid hourly targets:", hourly.temperature.notna().sum())
display(hourly.head())


# %% [markdown]
# ## Code Cell 2 — 24-hour-ahead lag features and strict time split
#
# Lags are constructed **before** the chronological split, so the earliest 2016 rows can use genuinely earlier 2015 observations. They never use the temperature being predicted.
#
# We use a common complete-case subset across models to keep their validation/test rows comparable. Missing or sentinel readings are dropped rather than filled using future values.
#

# %%
LAG_HOURS = [24, 48, 72, 168]
for lag in LAG_HOURS:
    hourly[f"lag_{lag}h"] = hourly["temperature"].shift(lag)

columns = ["temperature"] + [f"lag_{lag}h" for lag in LAG_HOURS]
samples = hourly.dropna(subset=columns).copy()

year = samples.index.year
train = year <= 2014
val = year == 2015
test = year == 2016

if not (train.any() and val.any() and test.any()):
    raise ValueError(
        "Expected observations before 2015, in 2015, and in 2016. "
        "Check the downloaded Gist and the available-years printout."
    )

y = samples["temperature"].to_numpy()
print("Training / validation / test:", train.sum(), val.sum(), test.sum())


# %% [markdown]
# ## Code Cell 3 — Sine and cosine basis functions
#
# A single sinusoid with unknown phase can be written as
#
# \[
# \hat y = \beta_0+\beta_1\sin(\omega x)+\beta_2\cos(\omega x).
# \]
#
# For hourly weather data, one sine wave is not enough: there are **annual** and **daily** cycles. Fourier basis functions stack the first few harmonics of each cycle. Products of annual and daily terms allow the daily cycle's shape/amplitude to change with season.
#
# The resulting model is still **linear in its coefficients**, so we can fit the features with ordinary least squares or Ridge:
#
# \[
# \hat y=\beta_0+\sum_j\beta_j\phi_j(t),\qquad
# L(\beta)=\sum_i(y_i-\hat y_i)^2+\alpha\sum_j\beta_j^2.
# \]
#
# The last few features may be historical temperatures: \(y_{t-24}\), \(y_{t-48}\), etc.
#

# %%
def fourier_features(frame, annual_h=2, daily_h=2,
                     interactions=True, lags=()):
    idx = frame.index
    annual_phase = 2 * np.pi * (
        (idx.dayofyear.to_numpy() - 1)
        + idx.hour.to_numpy() / 24
    ) / 365.25
    daily_phase = 2 * np.pi * (
        idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    ) / 24

    annual = np.column_stack([
        f(k * annual_phase)
        for k in range(1, annual_h + 1)
        for f in (np.sin, np.cos)
    ])
    daily = np.column_stack([
        f(k * daily_phase)
        for k in range(1, daily_h + 1)
        for f in (np.sin, np.cos)
    ])

    blocks = [annual, daily]
    if interactions:
        blocks.append(
            (annual[:, :, None] * daily[:, None, :])
            .reshape(len(frame), -1)
        )
    if lags:
        blocks.append(
            frame[[f"lag_{lag}h" for lag in lags]].to_numpy()
        )
    return np.column_stack(blocks)

def regression_scores(actual, prediction):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
        "MAE": float(mean_absolute_error(actual, prediction)),
        "R2": float(r2_score(actual, prediction))
    }

# Start with a simple annual + daily sine/cosine model.
X_basic = fourier_features(samples, annual_h=1, daily_h=1,
                           interactions=False, lags=())
ordinary = LinearRegression().fit(X_basic[train], y[train])
ridge_basic = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
ridge_basic.fit(X_basic[train], y[train])

print("Sine + OLS:", regression_scores(y[val], ordinary.predict(X_basic[val])))
print("Sine + Ridge:", regression_scores(y[val], ridge_basic.predict(X_basic[val])))


# %% [markdown]
# ## Code Cell 4 — Tune the Ridge model **on 2015 only**
#
# Try several annual/daily Fourier harmonic counts, season×hour interactions, 24/48-hour and longer **past** lags, and L2 penalties. The standardizer is fitted on the **training period only**, inside the pipeline; this avoids scaling leakage.
#
# **Do not adjust your settings using the 2016 test results.**
#

# %%
from itertools import product

feature_settings = [
    (year_h, day_h, interaction, lags)
    for (year_h, day_h), interaction, lags in product(
        [(1, 1), (2, 2), (3, 3), (2, 3)],
        [False, True],
        [(), (24, 48), (24, 48, 72, 168)]
    )
]
alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

records = []
for year_h, day_h, interaction, lags in feature_settings:
    X = fourier_features(
        samples, year_h, day_h,
        interactions=interaction, lags=lags
    )
    for alpha in alpha_grid:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[train], y[train])
        score = regression_scores(y[val], model.predict(X[val]))
        records.append({
            "annual_h": year_h, "daily_h": day_h,
            "interactions": interaction, "lags": lags,
            "alpha": alpha, "val_RMSE": score["RMSE"],
            "val_MAE": score["MAE"], "val_R2": score["R2"]
        })

results = pd.DataFrame(records).sort_values("val_RMSE").reset_index(drop=True)
display(results.head(10))

best_ridge = results.iloc[0].to_dict()
print("Best Ridge (selected using validation only):", best_ridge)


# %% [markdown]
# ## Code Cell 5 — Optional nonlinear contender for better temperature predictions
#
# Ridge is the focus of this session. If **maximizing predictive performance** is more important than keeping a purely linear model, test a histogram gradient-boosted tree using the same engineered features. It can learn nonlinear effects that a finite Fourier/Ridge model might miss. This is a distinct model, not Ridge.
#
# We only compare its **2015 validation RMSE** with Ridge before choosing a final model.
#

# %%
cfg = best_ridge
X_best = fourier_features(
    samples,
    annual_h=int(cfg["annual_h"]),
    daily_h=int(cfg["daily_h"]),
    interactions=bool(cfg["interactions"]),
    lags=tuple(cfg["lags"])
)

boost_search = []
for leaves, learning_rate in product([15, 31], [0.05, 0.1]):
    boost = HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=leaves,
        learning_rate=learning_rate, l2_regularization=1.0,
        random_state=42
    )
    boost.fit(X_best[train], y[train])
    scores = regression_scores(y[val], boost.predict(X_best[val]))
    boost_search.append({
        "leaves": leaves, "learning_rate": learning_rate,
        "val_RMSE": scores["RMSE"]
    })

boost_results = pd.DataFrame(boost_search).sort_values("val_RMSE")
display(boost_results)
best_boost = boost_results.iloc[0].to_dict()
winner = "Boosting" if best_boost["val_RMSE"] < best_ridge["val_RMSE"] else "Ridge"
print("Selected using 2015 validation:", winner)


# %% [markdown]
# ## Code Cell 6 — Refit on 2012–2015, evaluate untouched 2016 **once**
#
# The final chosen model is selected by **2015 validation RMSE**. After selection, retrain on 2012–2015 and evaluate on the 2016 holdout. The baseline and the two candidate families are reported for learning purposes; do not keep experimenting based on their test scores.
#

# %%
train_val = train | val
score_rows = []
predictions = {}

# The original simple Fourier/ordinary least-squares baseline.
simple = LinearRegression().fit(X_basic[train_val], y[train_val])
predictions["Sine (OLS)"] = simple.predict(X_basic[test])

# Selected Ridge configuration.
ridge_final = make_pipeline(StandardScaler(), Ridge(alpha=float(cfg["alpha"])))
ridge_final.fit(X_best[train_val], y[train_val])
predictions["Ridge + Fourier/lags"] = ridge_final.predict(X_best[test])

# Selected boosted-tree hyperparameters from 2015 validation.
boost_final = HistGradientBoostingRegressor(
    max_iter=200, max_leaf_nodes=int(best_boost["leaves"]),
    learning_rate=float(best_boost["learning_rate"]),
    l2_regularization=1.0, random_state=42
)
boost_final.fit(X_best[train_val], y[train_val])
predictions["Gradient boosting"] = boost_final.predict(X_best[test])

for name, pred in predictions.items():
    score_rows.append({"Model": name, **regression_scores(y[test], pred)})

final_scores = pd.DataFrame(score_rows).set_index("Model").round(3)
display(final_scores)
print("Chosen model (from validation only):", winner)


# %% [markdown]
# ## Code Cell 7 — Plot the 2016 forecast and two weeks of July
#
# These plots compare the observed hourly temperatures with the models. The seasonal sine-only model should be smooth; adding lag observations helps adapt to unusual short-term weather patterns.
#
# **Interpretation:** Lagged features at timestamp (t) are observations already measured by (t-24) hours or earlier. This evaluates rolling forecasts with past observations available, not a once-only prediction for the entire following year.
#

# %%
test_times = samples.index[test]
observed = pd.Series(y[test], index=test_times, name="Actual")
fig, axes = plt.subplots(2, 1, figsize=(13, 8))

axes[0].plot(observed.index, observed.values, color="gray",
             linewidth=0.5, alpha=0.6, label="Actual")
for name, pred in predictions.items():
    axes[0].plot(test_times, pred, linewidth=1.0, label=name)
axes[0].set_title("Yosemite — held-out 2016 temperatures")
axes[0].set_ylabel("Temperature (°C)")
axes[0].legend()

window = (test_times >= "2016-07-01") & (test_times < "2016-07-15")
axes[1].plot(test_times[window], y[test][window],
             color="gray", linewidth=2, label="Actual")
for name, pred in predictions.items():
    axes[1].plot(test_times[window], pred[window], label=name)
axes[1].set_title("Yosemite — July 1–14, 2016")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Reading the results and further improvements
#
# - **RMSE** is in °C, and smaller is better. **MAE** is the mean absolute error in °C. **R²** expresses improvement relative to predicting the test-set mean and is better when closer to 1.
# - **Fourier basis + Ridge** tests the session's regularization idea: the solution stays linear in basis coefficients even though the fitted curve is nonlinear in clock time.
# - **Lagged temperatures** are highly informative because temperatures are temporally correlated. They support a rolling day-ahead forecast but require actual recent observations at prediction time.
# - **Gradient boosting** may do better by learning nonlinear interactions. It is included as a separate benchmark, not described as Ridge.
# - The earlier class screenshot reported **RMSE 4.848 / R² 0.611** for season×hour RBF and **RMSE 2.910 / R² 0.860** for RBF with 24h/48h lags. Those are **from the earlier exercise**, not claimed as outputs of this notebook. They are not necessarily directly comparable unless splits and preprocessing match.
# - This notebook **does not tune on 2016**. If you want to improve further without contamination, use rolling-origin cross-validation within 2012–2015, inspect error by season/hour, and only then perform a final evaluation on 2016.
#
# To run locally, install `numpy pandas matplotlib scikit-learn jupyter`, open the notebook, and run all seven cells. The Gist download requires an internet connection.
#

