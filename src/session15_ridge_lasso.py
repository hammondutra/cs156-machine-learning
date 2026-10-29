"""CS156 Session 15 — Basis Functions 2: Ridge and Lasso.
Converted from the seven code cells in the matching notebook.
# %% blocks support interactive execution in VS Code.
"""

# %% [markdown]
# # CS156 — Session 15: Basis Functions 2 — Ridge and Lasso
#
# **Source:** CS156 Session 15 PCW, *Ridge and Lasso*, supplied 29 October 2026. Class title: [Basis Functions 2: Ridge and Lasso](https://forum.minerva.edu/app/courses/4077/sections/13535/classes/105085).
#
# This notebook recreates **the seven PCW code cells**, embeds answers/reflections from Questions 1–7, and makes minor compatibility changes for recent NumPy, JAX, and scikit-learn releases. The analysis uses a synthetic logistic-regression problem with true parameters \(\beta_0=-5\) and \(\beta_1=3\). No external dataset is required.
#
# **Run order:** execute cells from top to bottom. The three loss plots are the unregularized negative log-likelihood (blue), squared L2 norm (gray), and regularized objective (purple). For the sampling experiment, regenerate data and rerun the plotting cell, or run the 50-sample extension below.
#

# %% [markdown]
# ## Question 1 — Why regularize? Ridge vs. Lasso
#
# An ill-posed estimation problem may have no unique solution or may change drastically under small perturbations in the training sample. Regularization discourages unnecessarily large coefficients and can stabilize estimates, reducing overfitting.
#
# A linear prediction model commonly fits its parameters by minimizing the squared-error loss \(\sum_i(y_i-\hat y_i)^2\). **Ridge (L2)** adds \(\lambda\sum_j\beta_j^2\), shrinking weights toward zero; **Lasso (L1)** adds \(\lambda\sum_j|\beta_j|\), which can drive some weights exactly to zero and therefore select features.
#
# **Follow-up “why?” answers (from the PCW discussion):**
#
# - Large weights can make predictions more sensitive to small feature changes, particularly when the features are on comparable scales.
# - L1's absolute-value penalty has a corner at zero, making an exactly zero coefficient optimal in some settings.
# - With redundant features, many parameter combinations may predict equally well; a strictly positive Ridge penalty can select a unique smaller-norm solution.
# - Large coefficients are not inherently wrong: the penalty expresses a tradeoff against data fit.
#
# For this notebook's **logistic** model, the data-fitting loss is *average negative log-likelihood* (not squared error). The exercise penalizes both \(\beta_0\) and \(\beta_1\) in its manual L2 visualization.
#

# %% [markdown]
# ## Core question — Return to Bollywood
#
# We revisit the synthetic Bernoulli/logistic model from the maximum-likelihood session. Different random samples give different parameter estimates. The aim is to visualize how adding an L2 penalty changes the minimizing parameter pair and then quantify sampling variability.
#
# ### Code Cell 1 of 7 — Plotting and synthetic-data helpers
#
# The original PCW installed a Python 3.8-specific JAX wheel in this cell. Installation is **not performed automatically** here: use the repository `requirements.txt` and your own Python environment. The helper preserves the PDF's 20×20 parameter grid and its plotting ranges.
#

# %%
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sklearn
from sklearn.linear_model import LogisticRegression, RidgeClassifier

def plot_mesh(fig, ax, fn, x_data, y_data,
              x_range=(-6, -2), y_range=(0, 4), N=20,
              cmap="Blues_r", show_true=True):
    beta0_s = np.linspace(*x_range, N)
    beta1_s = np.linspace(*y_range, N)
    mesh_b0, mesh_b1 = np.meshgrid(beta0_s, beta1_s)
    params = {"beta_0": mesh_b0.ravel(), "beta_1": mesh_b1.ravel()}

    batched_fn = jax.vmap(fn, in_axes=(0, None, None))
    mesh_y = np.asarray(batched_fn(params, x_data, y_data)).reshape(N, N)

    contour = ax.contourf(mesh_b0, mesh_b1, mesh_y, levels=50, cmap=cmap)
    fig.colorbar(contour, ax=ax)
    ax.set_xlabel("Beta0")
    ax.set_ylabel("Beta1")

    min_arg = np.argmin(mesh_y)
    min_x = mesh_b0.ravel()[min_arg]
    min_y = mesh_b1.ravel()[min_arg]
    ax.scatter(min_x, min_y, marker="x", label="Estimated Minimum")

    if show_true:
        ax.scatter(-5, 3, marker="x", label="True Value")

    ax.set_title(f"Estimate: beta0={min_x:.2f}, beta1={min_y:.2f}")
    ax.legend()
    return min_x, min_y

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def generate_data(x):
    # Data-generating parameters: beta_0 = -5, beta_1 = 3.
    probability = sigmoid(3 * x - 5)
    return np.random.uniform() < probability

def generate_dataset(N):
    x_data = np.random.uniform(high=3, size=N)
    y_data = np.array([generate_data(x) for x in x_data], dtype=np.float32)
    return x_data, y_data


# %% [markdown]
# ### Code Cell 2 of 7 — Sample 1,000 observations
#
# The seed is an added reproducibility aid. The source PCW did not fix one, so its numerical minima and variances will generally differ from fresh runs.
#

# %%
np.random.seed(42)
NUM_DATAPOINTS = 1000
x_data, y_data = generate_dataset(NUM_DATAPOINTS)
print("Observations:", len(x_data), "| positive class:", int(y_data.sum()))


# %% [markdown]
# ### Code Cell 3 of 7 — Unregularized negative log-likelihood
#
# Maximizing the Bernoulli log-likelihood is equivalent to minimizing its negative. We use the *mean* rather than the sum of per-observation values; dividing by \(N\) does not change the optimum. A small probability clip prevents \(\log(0)\).
#

# %%
@jax.jit
def nll(params, xs, ys):
    logit = params["beta_0"] + params["beta_1"] * xs
    probs = jax.nn.sigmoid(logit)
    probs = jnp.clip(probs, 1e-7, 1 - 1e-7)
    likelihood = ys * jnp.log(probs) + (1 - ys) * jnp.log(1 - probs)
    return -jnp.mean(likelihood)

fig, ax = plt.subplots()
nll_minimum = plot_mesh(fig, ax, nll, x_data, y_data, cmap="Blues_r")
plt.show()


# %% [markdown]
# ### Code Cell 4 of 7 — Squared L2 norm
#
# The prompt calls this the “L2 norm,” but the supplied implementation is its **square**, \(\|\beta\|_2^2=\beta_0^2+\beta_1^2\), as used for Ridge regularization. It is a parameter penalty, not a squared *prediction error*.
#

# %%
@jax.jit
def l2_norm(params, xs, ys):
    beta_0 = params["beta_0"]
    beta_1 = params["beta_1"]
    norm = beta_0**2 + beta_1**2
    return norm

fig, ax = plt.subplots()
l2_minimum = plot_mesh(
    fig, ax, l2_norm, x_data, y_data, show_true=False, cmap="Greys_r"
)
plt.show()


# %% [markdown]
# ## Questions 2–4 — Interpreting the gray L2 surface
#
# **Q2 — What is shown?** The horizontal and vertical axes are \(\beta_0\) and \(\beta_1\). The grayscale values encode \(\beta_0^2+\beta_1^2\); darker regions correspond to smaller squared norms.
#
# **Q3 — Which minimum?** Within the plotted bounds \(\beta_0\in[-6,-2]\), \(\beta_1\in[0,4]\), the minimum occurs at **(−2, 0)**, with squared norm **4**. This is a boundary-constrained minimum. Over all \(\mathbb R^2\), the global minimum is **(0, 0)**, where the penalty is zero.
#
# **Q4 — How does the scale compare?** On the PCW grid, the penalty takes values about **4–52**, whereas average NLL values are considerably smaller. Adding them without a scaling coefficient would cause the squared norm to dominate the fitted objective. The coefficient \(\alpha\) controls how strongly the penalty influences the fitted parameters.
#

# %% [markdown]
# ### Code Cell 5 of 7 — Add L2 regularization
#
# Here \(L(\beta)=\mathrm{NLL}(\beta)+\alpha\|\beta\|_2^2\), with \(\alpha=10^{-2}\). Uncomment the indicated line and rerun this cell **five times** to explore sensitivity to new datasets, as requested in Question 5.
#

# %%
@jax.jit
def modified_loss(params, xs, ys, alpha=1e-2):
    return nll(params, xs, ys) + alpha * l2_norm(params, xs, ys)

# Uncomment to resample when asked; rerun this cell five times.
# x_data, y_data = generate_dataset(NUM_DATAPOINTS)

fig, ax = plt.subplots(ncols=3, figsize=(14, 3))
original_minimum = plot_mesh(fig, ax[0], nll, x_data, y_data, cmap="Blues_r")
penalty_minimum = plot_mesh(
    fig, ax[1], l2_norm, x_data, y_data, show_true=False, cmap="Greys_r"
)
regularized_minimum = plot_mesh(
    fig, ax[2], modified_loss, x_data, y_data, cmap="Purples_r"
)
plt.tight_layout()
plt.show()
print("Original minimum (grid):", original_minimum)
print("Regularized minimum (grid):", regularized_minimum)


# %% [markdown]
# ## Question 5 — Shift and resampling
#
# The **original blue** minimum lies near the true parameter pair \((-5,3)\) (the supplied PDF's example was around \((-5.16,3.16)\)). The **modified purple** minimum is shifted toward smaller coefficient magnitudes (the PDF's example was around \((-2.21,1.47)\)). The gray penalty independently prefers \((-2,0)\) inside the plotted rectangle.
#
# This shift occurs because the optimized function now contains an additional term favoring the origin: \(\mathrm{NLL}(\beta)+\alpha\|\beta\|_2^2\). Under repeated resampling, regularized parameter estimates are expected to vary less than unregularized estimates, at the expense of bias. **The five additional trial minima were not recorded in the submitted PCW**, so they are not invented here; run the resampling cell to collect your own five trials.
#

# %% [markdown]
# ## Extension 4 — Thinking in distributions
#
# Generate **50** datasets. On each one, fit an **unregularized** `LogisticRegression` and a `RidgeClassifier(alpha=1e-2)`; store intercept and slope, scatter their 50 parameter pairs, and compute sample variance for each coefficient.
#
# **Compatibility notes:** scikit-learn accepts `penalty="none"` in older versions and `penalty=None` in many newer versions; modern releases prefer `C=np.inf`. Also, depending on scikit-learn version, `RidgeClassifier.coef_` may be 1D or 2D, so we flatten it safely.
#
# **Comparison caveat:** `RidgeClassifier` minimizes a different data-fit loss from logistic regression and usually does **not** penalize its intercept. Thus the experiment is an illustrative variance comparison, not a controlled isolation of the regularization effect from objective-function differences.
#
# ### Code Cell 6 of 7 — Repeated model estimation
#

# %%
logistic_params = []
ridge_params = []

version = tuple(int(v) for v in sklearn.__version__.split(".")[:2])

for i in range(50):
    x_data, y_data = generate_dataset(NUM_DATAPOINTS)
    X = x_data.reshape(-1, 1)

    # Match the PDF's penalty-free LogisticRegression across sklearn versions.
    if version >= (1, 8):
        logistic = LogisticRegression(C=np.inf, max_iter=1000)
    else:
        penalty_value = None if version >= (1, 2) else "none"
        logistic = LogisticRegression(penalty=penalty_value, max_iter=1000)
    ridge = RidgeClassifier(alpha=1e-2)

    logistic.fit(X, y_data)
    ridge.fit(X, y_data)

    logistic_params.append([
        float(np.asarray(logistic.intercept_).ravel()[0]),
        float(np.asarray(logistic.coef_).ravel()[0])
    ])
    ridge_params.append([
        float(np.asarray(ridge.intercept_).ravel()[0]),
        float(np.asarray(ridge.coef_).ravel()[0])
    ])

logistic_params = np.array(logistic_params)
ridge_params = np.array(ridge_params)
print("Fitted models per class:", len(logistic_params))


# %% [markdown]
# ### Code Cell 7 of 7 — Scatter plots and coefficient variances
#
# The PCW text mentions “10 coefficient estimates” once, but its instructions and starter loop specify **50** runs; the implementation uses all 50.
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, params, title in zip(
    axes,
    [logistic_params, ridge_params],
    ["Logistic Regression", "Ridge Classifier"]
):
    ax.scatter(params[:, 0], params[:, 1], alpha=0.7)
    ax.set_xlabel("Beta 0 (Intercept)")
    ax.set_ylabel("Beta 1 (Coefficient)")
    ax.set_title(title)

plt.tight_layout()
plt.show()

print("Logistic Regression variance:", np.var(logistic_params, axis=0, ddof=1))
print("Ridge Classifier variance:", np.var(ridge_params, axis=0, ddof=1))


# %% [markdown]
# ## Question 6 — Which model has higher parameter variance?
#
# Qualitatively, the submitted PCW's **unregularized logistic-regression** parameter cloud is considerably more dispersed than the `RidgeClassifier` cloud. Quantitatively, the **submitted PDF's one 50-run output** reports approximate sample variances:
#
# | Model | Var(β₀) | Var(β₁) |
# |---|---:|---:|
# | Unregularized LogisticRegression | 0.11753137 | 0.03530237 |
# | RidgeClassifier (alpha=0.01) | 0.00102374 | 0.00033064 |
#
# These are **recorded source-PCW values**, not guaranteed output from a new run. Since the estimators use different likelihood/loss formulations and may handle the intercept differently, the difference cannot be attributed to regularization alone.
#
# ## Question 7 — Bias–variance tradeoff
#
# Bias is the systematic error introduced by a model's assumptions; variance is the sensitivity of its estimate to the particular training sample. Greater flexibility often lowers bias but increases variance and the risk of overfitting. Regularization typically increases bias while lowering variance, helping models generalize better. The goal is **lower expected error on unseen data**, not merely minimum training loss.
#
# ## Question 8 — Upload slot
#
# The PCW's final question is simply an optional file-upload slot; no additional computational exercise was provided.
#
# **Practical note:** The source session focuses experimentally on **L2/Ridge** regularization. L1/Lasso appears in the introductory conceptual comparison; no standalone Lasso fitting experiment is present in the PDF.
#
