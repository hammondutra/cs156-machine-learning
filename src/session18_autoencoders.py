"""CS156 Session 18 — autoencoders, deeper bottlenecks, and teacher forcing.

Runnable version of the Session 18 PCW exercises.
Run: python src/session18_autoencoders.py
Optional quick check: CS156_STEPS=100 CS156_TEACHER_STEPS=50 CS156_NO_SHOW=1 python src/session18_autoencoders.py
"""

# %% [markdown]
# # CS156 — Session 18: Autoencoders
# 
# **Source:** supplied *CS156 Session 18 PreClassWork* PDF (10 pages, 10 November 2026). This is a cleaned, executable adaptation of its **Questions 1–4** and the **extension and super-extension**. Question 5 is a PDF-upload field, not an additional modeling exercise.
# 
# **Learning goals:** representation learning, encoder/decoder networks, two-dimensional bottlenecks, nonlinear dimensionality reduction, reconstruction loss, backpropagation with JAX and Adam, visualization of handwritten digits, deeper-network comparisons, and autoregressive teacher forcing.
# 
# **Data:** scikit-learn's built-in 8 × 8 handwritten digits, 1,797 images, 64 intensity features normalized into [0, 1]. No external dataset download is needed. Labels are used only to color the latent-space plot, never to optimize the autoencoder.
# 
# **Dependencies:** NumPy, Matplotlib, scikit-learn, JAX, and Optax. Install the repository requirements. Run this notebook from top to bottom.
# 
# **Performance:** full runs use 3,000 optimizer steps for each feedforward training experiment plus 1,000 for the optional teacher-forced model. For a quick run, set environment variables **CS156_STEPS=100** and **CS156_TEACHER_STEPS=50** before launching Jupyter or Python. The short run is for checking functionality, not reproducing source scores.
# 
# ## Question 1 of 5 — Autoencoder introduction and LLM prompts
# 
# **Original prompts from the PCW:**
# 1. Give a high-level explanation of how an autoencoder works.
# 2. Give a detailed, step-by-step explanation of how an autoencoder produces a compressed representation of its input.
# 
# **High-level explanation:** An autoencoder trains an **encoder** to transform an input into a smaller latent vector, and a **decoder** to reconstruct the original input from that vector. The target is the original input itself; class labels are unnecessary. A bottleneck constrains what information can pass through, encouraging the encoder and decoder to preserve useful structure.
# 
# **Mechanism:** (1) Normalize the pixels; (2) apply learned affine transformations and nonlinear activations in the encoder; (3) represent each image with just two real numbers; (4) use the decoder to predict all 64 original pixels; (5) compute reconstruction cross-entropy; (6) backpropagate through both networks and update weights; (7) repeat until the model learns a useful compromise between compression and reconstruction. The two coordinates need not correspond to human-labeled traits.
# 
# **Follow-up from the PCW (under 150 words):** Nonlinear autoencoders use nonlinear activation functions to learn lower-dimensional representations that need not be linear projections of the inputs. PCA chooses a linear subspace, whereas a nonlinear encoder and decoder can model curved structures and more complex relationships. Nonlinear autoencoders are therefore **potentially more expressive than PCA**, although they require iterative training, can overfit, and may not produce a more useful representation for every dataset. Under appropriate conditions a linear autoencoder trained with squared reconstruction error learns the same principal subspace as PCA.
# 
# **Source reflection:** The submitted critique said the response was informative but did not provide a substantive evaluation. An improved critique would explicitly distinguish *greater representational capacity* from *better generalization* and point out that PCA's linear solution is simpler, more interpretable, and often more data-efficient.
# 
# ## Question 2 — Implement an autoencoder for handwritten digits
# 
# The source PCW has **five trainable affine layers**. Its architecture is **64 → 8 → 8 → 2 → 8 → 64**. The encoder reaches the 2D bottleneck in layer 3; the decoder expands it in layers 4–5. The bottleneck is **linear**, not passed through ReLU. The output uses sigmoid to produce pixel values between zero and one. The return dictionary exposes both **reconstruct** (for the loss) and **bottleneck** (for visualization).
# 
# ### Code Cell 1 — Imports and experiment configuration

# %%
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from sklearn import datasets

STEPS = int(os.getenv("CS156_STEPS", "3000"))
TEACHER_STEPS = int(os.getenv("CS156_TEACHER_STEPS", "1000"))
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

def finish_plot():
    if os.getenv("CS156_NO_SHOW") == "1":
        plt.close()
    else:
        plt.show()

print("JAX devices:", jax.devices())


# %% [markdown]
# ### Code Cell 2 — Load and normalize the digits dataset
# 
# The images are 8 × 8 grayscale arrays. The original digit intensities range from 0 to 16; dividing by 16 produces numbers in [0, 1]. The **y_data** values are used only to interpret the learned embedding.

# %%
digits = datasets.load_digits()
x_data = digits["data"] / 16.0
y_data = digits["target"]
X = jnp.asarray(x_data, dtype=jnp.float32)
print("Image matrix:", X.shape, "| Labels:", y_data.shape)


# %% [markdown]
# ### Code Cell 3 — Visualize one original digit

# %%
plt.figure(figsize=(4, 4))
plt.imshow(x_data[0].reshape(8, 8), cmap="Greys", vmin=0, vmax=1)
plt.colorbar(label="Normalized pixel intensity")
plt.title(f"Original digit (label: {y_data[0]})")
finish_plot()


# %% [markdown]
# ### Code Cell 4 — Initialize the original five-layer network
# 
# Each **W** is a weight matrix and each **b** is a bias vector. The hidden layers use ReLU, the central bottleneck remains linear, and the output uses sigmoid.
# 
# The training objective is **binary cross-entropy on pixel intensities**, following the supplied PCW. This is a reconstruction objective rather than a digit-classification objective.

# %%
def initialize_layers(dimensions, seed=42):
    rng = np.random.default_rng(seed)
    return {
        f"layer_{i}": {
            "W": jnp.asarray(
                rng.normal(size=(n_in, n_out)) / np.sqrt(n_in),
                dtype=jnp.float32
            ),
            "b": jnp.zeros((n_out,), dtype=jnp.float32),
        }
        for i, (n_in, n_out) in enumerate(
            zip(dimensions[:-1], dimensions[1:]), start=1
        )
    }


def forward_layers(parameters, inputs):
    h = jnp.asarray(inputs, dtype=jnp.float32)
    layer_count = len(parameters)
    middle_layer = (layer_count + 1) // 2

    for i in range(1, layer_count + 1):
        layer = parameters[f"layer_{i}"]
        h = h @ layer["W"] + layer["b"]
        if i == middle_layer:
            bottleneck = h  # Linear 2D latent representation
        elif i == layer_count:
            h = jax.nn.sigmoid(h)
        else:
            h = jax.nn.relu(h)

    return {"reconstruct": h, "bottleneck": bottleneck}


def reconstruction_loss(reconstructed, targets):
    reconstructed = jnp.clip(reconstructed, 1e-7, 1 - 1e-7)
    return -jnp.mean(
        targets * jnp.log(reconstructed)
        + (1 - targets) * jnp.log1p(-reconstructed)
    )


def loss_fn(parameters, inputs):
    outputs = forward_layers(parameters, inputs)
    return reconstruction_loss(outputs["reconstruct"], inputs)


params = initialize_layers([64, 8, 8, 2, 8, 64], seed=42)
print("Five-layer model initialized.")


# %% [markdown]
# ### Code Cell 5 — Train the original model with backpropagation
# 
# An **Adam** optimizer adjusts both encoder and decoder weights. Each gradient is the derivative of the reconstruction loss with respect to the model parameters. The source PCW trained for 3,000 steps and ended at approximately **0.379** cross-entropy in its initial run; exact results depend on initialization and minibatch order.

# %%
optimizer = optax.adam(LEARNING_RATE)

@jax.jit
def train_step(parameters, state, batch):
    gradients = jax.grad(loss_fn)(parameters, batch)
    updates, state = optimizer.update(gradients, state)
    return optax.apply_updates(parameters, updates), state

state = optimizer.init(params)
rng_original = np.random.default_rng(17)
original_history = []
for step in range(STEPS):
    indices = rng_original.integers(0, X.shape[0], size=BATCH_SIZE)
    params, state = train_step(params, state, X[indices])
    if step % 100 == 0 or step == STEPS - 1:
        loss_value = float(loss_fn(params, X))
        original_history.append((step, loss_value))
        print(f"Step {step:4d} | loss {loss_value:.6f}")

print(f"Original five-layer final loss: {float(loss_fn(params, X)):.6f}")


# %% [markdown]
# ## Question 3 — Reconstruction quality
# 
# The source PCW reported reconstructions that often preserved the digit's general shape but were **blurrier** than the originals. Some digits were difficult to recognize. A 2D bottleneck is a severe constraint: it must summarize an entire 64-pixel image using only two values.
# 
# ### Code Cell 6 — Obtain reconstructions and bottleneck coordinates

# %%
predicted = forward_layers(params, X)
reconstructed_digits = np.asarray(predicted["reconstruct"])
latent_points = np.asarray(predicted["bottleneck"])
print("Reconstructed pixels:", reconstructed_digits.shape)
print("Latent coordinates:", latent_points.shape)


# %% [markdown]
# ### Code Cell 7 — Compare a reconstructed image with its original
# 
# Set the integer **EXAMPLE_NUMBER** to any valid index from 0 through 1796 to inspect other examples.

# %%
EXAMPLE_NUMBER = 28
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(x_data[EXAMPLE_NUMBER].reshape(8, 8), cmap="Greys", vmin=0, vmax=1)
axes[0].set_title("Original")
axes[1].imshow(reconstructed_digits[EXAMPLE_NUMBER].reshape(8, 8), cmap="Greys", vmin=0, vmax=1)
axes[1].set_title("Reconstructed")
for ax in axes:
    ax.axis("off")
plt.suptitle(f"Example {EXAMPLE_NUMBER} (label {y_data[EXAMPLE_NUMBER]})")
plt.tight_layout()
finish_plot()


# %% [markdown]
# ## Question 4 — Interpret the two-dimensional bottleneck
# 
# **Question:** Are examples of the same digit close to one another in the 2D plot?
# 
# **Observation from the supplied PCW:** some digits form recognizable concentrations, but clusters often overlap. The author's notes especially called out distinct groups for some digits and substantial overlap for others.
# 
# **Why is this possible without labels?** The autoencoder is trained only to reconstruct its own input. Similar handwriting can induce similar latent representations, so digit-related grouping can emerge **without supervised labels**. However, reconstruction does not explicitly require class separation, and overlap is expected.
# 
# ### Code Cell 8 — Plot the two-dimensional embedding

# %%
fig, ax = plt.subplots(figsize=(8, 6))
points = ax.scatter(
    latent_points[:, 0], latent_points[:, 1],
    c=y_data, cmap="tab10", s=12, alpha=0.65
)
colorbar = fig.colorbar(points, ax=ax, ticks=range(10))
colorbar.set_label("True digit label (not used in training)")
ax.set_xlabel("Latent dimension 1")
ax.set_ylabel("Latent dimension 2")
ax.set_title("Unsupervised 2D digit representations")
finish_plot()


# %% [markdown]
# ## Extension — A better (deeper) autoencoder
# 
# The extension adds **one encoder layer and one decoder layer**, increasing the affine-layer count from 5 to 7:
# 
# - Original: **64 → 8 → 8 → 2 → 8 → 64**.
# - Deeper: **64 → 8 → 8 → 8 → 2 → 8 → 8 → 64**.
# 
# The bottleneck stays 2-dimensional, and both networks use the **same linear bottleneck, ReLU hidden activations, sigmoid output, Adam, loss, minibatches, and step count**. This gives a controlled architectural comparison, though a single seed and in-sample loss cannot establish consistent superiority.
# 
# ### Code Cell 9 — Initialize the deeper network

# %%
params_deep = initialize_layers([64, 8, 8, 8, 2, 8, 8, 64], seed=42)
params_baseline = initialize_layers([64, 8, 8, 2, 8, 64], seed=42)

print("Baseline layers:", len(params_baseline))
print("Deeper layers:", len(params_deep))
print("Both bottlenecks have dimension 2.")


# %% [markdown]
# ### Code Cell 10 — Retrain and compare five vs. seven layers
# 
# Both models see the **same sampled minibatch at each step**. We record full-dataset reconstruction loss every 100 steps for an apples-to-apples training comparison.
# 
# **Results reported in the supplied PDF, not guaranteed for every rerun:** 5-layer loss **0.374349**; 7-layer loss **0.374409**. The deeper model was marginally worse in that run (difference approximately 0.000060). More capacity does not guarantee improved optimization or reconstruction, especially with a strict bottleneck.
# 
# **Caution:** these are training-set losses, not held-out test performance.

# %%
state_baseline = optimizer.init(params_baseline)
state_deep = optimizer.init(params_deep)

rng_compare = np.random.default_rng(42)
history_baseline, history_deep, steps_recorded = [], [], []

for step in range(STEPS):
    indices = rng_compare.integers(0, X.shape[0], size=BATCH_SIZE)
    batch = X[indices]
    params_baseline, state_baseline = train_step(
        params_baseline, state_baseline, batch
    )
    params_deep, state_deep = train_step(
        params_deep, state_deep, batch
    )

    if step % 100 == 0 or step == STEPS - 1:
        steps_recorded.append(step)
        history_baseline.append(float(loss_fn(params_baseline, X)))
        history_deep.append(float(loss_fn(params_deep, X)))

baseline_loss = float(loss_fn(params_baseline, X))
deep_loss = float(loss_fn(params_deep, X))
print(f"Five-layer loss:  {baseline_loss:.6f}")
print(f"Seven-layer loss: {deep_loss:.6f}")
print(f"Improvement:      {baseline_loss - deep_loss:+.6f}")
print("Deeper model performed better:", deep_loss < baseline_loss)


# %% [markdown]
# ### Code Cell 11 — Plot learning curves and reconstructed digits
# 
# Inspect reconstruction details, not just a single scalar loss. Even if one architecture has a slightly lower loss, its output may not be perceptually better on every example.

# %%
plt.figure(figsize=(9, 4))
plt.plot(steps_recorded, history_baseline, label="5 layers")
plt.plot(steps_recorded, history_deep, label="7 layers")
plt.xlabel("Training step")
plt.ylabel("Full-dataset reconstruction loss")
plt.title("Autoencoder architecture comparison")
plt.grid(alpha=0.3)
plt.legend()
finish_plot()

baseline_images = np.asarray(forward_layers(params_baseline, X)["reconstruct"])
deep_images = np.asarray(forward_layers(params_deep, X)["reconstruct"])

examples = [0, 25, 100, 500, 1000]
fig, axes = plt.subplots(len(examples), 3, figsize=(7, 10))
for row, idx in enumerate(examples):
    for col, image in enumerate([x_data[idx], baseline_images[idx], deep_images[idx]]):
        axes[row, col].imshow(np.asarray(image).reshape(8, 8), cmap="Greys", vmin=0, vmax=1)
        axes[row, col].axis("off")
for col, heading in enumerate(["Original", "5 layers", "7 layers"]):
    axes[0, col].set_title(heading)
plt.tight_layout()
finish_plot()


# %% [markdown]
# ## Super Extension — Teacher forcing
# 
# **Teacher forcing does not apply directly to the original parallel, feedforward decoder.** We instead replace it with an **autoregressive recurrent decoder** that predicts one pixel at a time. It takes the learned 2D latent vector and the previous pixel.
# 
# During teacher-forced training, the model receives the **true previous pixel**. During generation, it receives its **own previous prediction**. As a result, teacher-forced loss and autonomous/free-running reconstruction loss measure different conditions.
# 
# This is an exploratory extension using raster pixel order, **not** a standard or necessarily superior image-autoencoder design.
# 
# ### Code Cell 12 — Build a recurrent decoder with teacher forcing

# %%
RNN_DIM = 32
rng_tf = np.random.default_rng(2026)

def random_weight(n_in, n_out):
    return jnp.asarray(
        rng_tf.normal(size=(n_in, n_out)) / np.sqrt(n_in),
        dtype=jnp.float32
    )

teacher_params = {
    "encoder": {
        f"layer_{i}": params_deep[f"layer_{i}"]
        for i in range(1, 5)
    },
    "decoder": {
        "W_prev": random_weight(1, RNN_DIM),
        "W_hidden": random_weight(RNN_DIM, RNN_DIM),
        "W_z": random_weight(2, RNN_DIM),
        "b_hidden": jnp.zeros(RNN_DIM),
        "W_out": random_weight(RNN_DIM, 1),
        "b_out": jnp.zeros(1),
    },
}

def encode_teacher(parameters, inputs):
    h = inputs
    for i in range(1, 5):
        layer = parameters["encoder"][f"layer_{i}"]
        h = h @ layer["W"] + layer["b"]
        if i < 4:
            h = jax.nn.relu(h)
    return h

def decode_teacher(parameters, z, true_pixels=None):
    d = parameters["decoder"]
    n = z.shape[0]
    hidden_zero = jnp.zeros((n, RNN_DIM))
    start_token = jnp.zeros((n, 1))

    def recurrent_step(hidden, previous_pixel):
        hidden = jnp.tanh(
            previous_pixel @ d["W_prev"]
            + hidden @ d["W_hidden"]
            + z @ d["W_z"] + d["b_hidden"]
        )
        output = jax.nn.sigmoid(hidden @ d["W_out"] + d["b_out"])
        return hidden, output

    if true_pixels is not None:
        previous = jnp.concatenate(
            [start_token, true_pixels[:, :-1]], axis=1
        ).T[:, :, None]

        def step(hidden, previous_pixel):
            hidden, output = recurrent_step(hidden, previous_pixel)
            return hidden, output[:, 0]

        _, sequence = jax.lax.scan(step, hidden_zero, previous)
    else:
        def step(carry, _):
            hidden, previous_pixel = carry
            hidden, output = recurrent_step(hidden, previous_pixel)
            return (hidden, output), output[:, 0]

        _, sequence = jax.lax.scan(
            step, (hidden_zero, start_token), None, length=64
        )
    return sequence.T

def teacher_loss(parameters, inputs):
    z = encode_teacher(parameters, inputs)
    reconstructed = decode_teacher(parameters, z, true_pixels=inputs)
    return reconstruction_loss(reconstructed, inputs)

teacher_optimizer = optax.adam(LEARNING_RATE)
teacher_state = teacher_optimizer.init(teacher_params)

@jax.jit
def teacher_train_step(parameters, state, batch):
    gradients = jax.grad(teacher_loss)(parameters, batch)
    updates, state = teacher_optimizer.update(gradients, state)
    return optax.apply_updates(parameters, updates), state


# %% [markdown]
# ### Code Cell 13 — Train and evaluate both decoding modes
# 
# **Reported in the source PDF:** teacher-forced loss **0.337815** versus free-running loss **0.442996**. This gap illustrates **exposure bias**: excellent one-step predictions conditioned on the correct past do not necessarily translate to good long-run generation.
# 
# The relevant metric for a self-contained autoencoder is the **free-running** reconstruction loss. These values depend on initialization and optimization, and they were obtained without a separate test split.

# %%
rng_teacher = np.random.default_rng(42)
for step in range(TEACHER_STEPS):
    indices = rng_teacher.integers(0, X.shape[0], size=BATCH_SIZE)
    teacher_params, teacher_state = teacher_train_step(
        teacher_params, teacher_state, X[indices]
    )

forced_loss = float(teacher_loss(teacher_params, X))
encoded = encode_teacher(teacher_params, X)
generated = decode_teacher(teacher_params, encoded)
free_running_loss = float(reconstruction_loss(generated, X))

print(f"Teacher-forced loss: {forced_loss:.6f}")
print(f"Free-running loss:   {free_running_loss:.6f}")
print("Compare free-running, not teacher-forced, loss with other autoencoders.")

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
idx = 28
images = [
    X[idx],
    decode_teacher(teacher_params, encoded, true_pixels=X)[idx],
    generated[idx],
]
for ax, img, heading in zip(
    axes, images, ["Original", "Teacher-forced", "Free-running"]
):
    ax.imshow(np.asarray(img).reshape(8, 8), cmap="Greys", vmin=0, vmax=1)
    ax.set_title(heading)
    ax.axis("off")
plt.tight_layout()
finish_plot()

