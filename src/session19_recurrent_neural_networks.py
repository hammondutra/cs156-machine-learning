"""CS156 Session 19 — batched vanilla RNN and Shakespeare GRU text generation.

Run: python src/session19_recurrent_neural_networks.py
Quick GPU/CPU check: CS156_EPOCHS=1 CS156_GRU_UNITS=64 CS156_MAX_BATCHES=2 CS156_GENERATE=25 CS156_NO_SHOW=1 python src/session19_recurrent_neural_networks.py
Requires TensorFlow and tf-keras; the Shakespeare dataset downloads on first use.
"""

# %% [markdown]
# # CS156 — Session 19: Recurrent Neural Networks (RNNs)
#
# **Topics:** recurrence, hidden states, batched sequence processing, recurrent weight sharing, backpropagation through time, vanishing/exploding gradients, LSTM and GRU gating, character embeddings, next-character prediction, and autoregressive Shakespeare-style text generation.
#
# **Source:** supplied *Session 19 PCW* (7-page submission capture, 12 November 2026) and the official [TensorFlow character-level RNN tutorial](https://www.tensorflow.org/text/tutorials/text_generation). Questions 1–4 cover RNN concepts and the NumPy code reproduced from the PCW. Questions 5–6 cover the Shakespeare GRU extension. Question 7 is only an optional PDF-output upload slot.
#
# **Notebook and Python parity:** The companion script in **src/session19_recurrent_neural_networks.py** contains the same explanations and executable experiments, organized using VS Code / Jupyter-style cell delimiters.
#
# **Running:** Install dependencies with \`pip install -r requirements.txt\`, then run cells in order, ideally on a Colab GPU for the Shakespeare section. The default follows the tutorial (20 epochs, 256 embedding dimensions, 1,024 GRU units, 1,000 generated characters); use the environment variables \`CS156_EPOCHS\`, \`CS156_GRU_UNITS\`, \`CS156_GENERATE\`, and \`CS156_MAX_BATCHES\` to shorten a local run. Training output is not fabricated or prefilled.

# %% [markdown]
# ## Question 1 of 7 — RNN and LSTM explanations
#
# ### Original prompts and polished responses
#
# **Prompt 1 (from the PCW conversation):** "Define a Recurrent Neural Network."
#
# A **recurrent neural network (RNN)** processes sequential inputs by retaining a **hidden state**, a learned representation of preceding inputs. At timestep $t$, it combines the current observation $x_t$ with the state $h_{t-1}$ from the previous timestep. The same input-to-hidden, hidden-to-hidden, and output weights are reused at every timestep.
#
# With **column vectors**, a vanilla RNN can be expressed as
#
# $$
# \begin{aligned}
# h_t &= \tanh(W_x x_t + W_h h_{t-1} + b_h),\\
# \hat y_t &= W_y h_t+b_y.
# \end{aligned}
# $$
#
# A large hidden state is *not* a perfect archive of earlier inputs: the network learns which history to encode, and long-range information may fade. Recurrence captures sequence dependence; it does not mix independent samples within a batch.
#
# **Suggested improvement to the original high-level response:** Emphasize both **shared parameters across time** and **the feedback of the prior hidden state**, rather than describing recurrence only as "memory." Distinguish sequence timesteps from network layers.

# %% [markdown]
# ### Prompt 2 — Step-by-step LSTM learning algorithm
#
# **Prompt:** "Explain how the LSTM learning algorithm works."
#
# An **LSTM** is a gated RNN with a hidden state $h_t$ and a **cell state** $c_t$. The gates are learned functions of the current input and previous hidden state:
#
# $$
# \begin{aligned}
# f_t &= \sigma(W_f [h_{t-1};x_t]+b_f) &&\text{(forget gate)},\\
# i_t &= \sigma(W_i [h_{t-1};x_t]+b_i) &&\text{(input gate)},\\
# \tilde c_t &= \tanh(W_c [h_{t-1};x_t]+b_c) &&\text{(candidate memory)},\\
# c_t &= f_t\odot c_{t-1}+i_t\odot\tilde c_t &&\text{(cell update)},\\
# o_t &= \sigma(W_o [h_{t-1};x_t]+b_o) &&\text{(output gate)},\\
# h_t &= o_t\odot\tanh(c_t) &&\text{(visible state)}.
# \end{aligned}
# $$
#
# Here $\sigma$ is the logistic sigmoid, $[h_{t-1};x_t]$ means concatenation, and $\odot$ means element-wise multiplication. The gates control information flow; **they are not themselves the optimization algorithm**.
#
# **How the weights are learned:**
#
# 1. **Forward:** Process the training sequence, keeping each intermediate gate, $c_t$, $h_t$, and prediction.
# 2. **Loss:** Compare predicted outputs to targets (for next-character prediction, usually categorical cross-entropy).
# 3. **Backpropagation through time (BPTT):** Apply the chain rule backward across the unrolled steps to differentiate the loss with respect to all shared gate and output weights.
# 4. **Optimization:** Update those weights using SGD or Adam; repeat for multiple batches and epochs.
#
# The additive cell update creates a direct pathway for gradients: holding the gates fixed, $\partial c_t/\partial c_{t-1}=f_t$. When components of $f_t$ are near one, their cell-state gradient can be preserved more effectively than in a vanilla RNN. LSTMs **mitigate**, but do not universally eliminate, vanishing or exploding gradients.
#
# **Suggested improvement:** Separate the architecture's forward gating operations from the actual learning procedure (BPTT plus optimizer), and use a consistent notation for all states and gates.

# %% [markdown]
# ### Follow-up (150 words or fewer): What makes an RNN recurrent?
#
# An RNN is **recurrent** because it repeatedly applies the *same learned transition function* along a sequence, passing its hidden state from one timestep into the next. The current state depends on both the current input and the preceding state:
#
# $$
# h_t=f(W_xx_t+W_hh_{t-1}+b).
# $$
#
# This creates a feedback connection **across time**, even though an unrolled RNN can be drawn as a chain of repeated computation blocks. Consequently, an output at time $t$ can depend on earlier inputs from the same sequence. Each element in a batch has its own state; separate sequences do not exchange information. In this context, "recurrent" does **not** mean that the RNN has to revisit the same training sample in multiple epochs.
#
# **Reflection:** The answer is appropriately concise; explicitly naming shared weights and state feedback makes the definition more precise.

# %% [markdown]
# ## Questions 2–4 — Implement a batched vanilla RNN
#
# The PCW uses **row-vector batches**, so its equations multiply on the right (a transpose of the column-vector convention above):
#
# $$
# \begin{aligned}
# H_t &= \tanh(X_tW_1+H_{t-1}W_2+b_1),\\
# Y_t &= H_tW_3+b_2.
# \end{aligned}
# $$
#
# This model is a **forward-pass demonstration**, not a trained predictor: all parameters are randomly initialized and no loss or gradient update is performed in Questions 2–4.
#
# ### Code Cell 1 — Reproduce the PCW RNN class

# %%
import os
import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.uniform(size=(input_dim, hidden_dim))
        self.W2 = np.random.uniform(size=(hidden_dim, hidden_dim))
        self.W3 = np.random.uniform(size=(hidden_dim, output_dim))
        self.b1 = np.random.uniform(size=(hidden_dim,))
        self.b2 = np.random.uniform(size=(output_dim,))

    def step(self, X, H):
        H = np.tanh(X @ self.W1 + H @ self.W2 + self.b1)
        Y = H @ self.W3 + self.b2
        return H, Y

# %% [markdown]
# ### Code Cell 2 — Set the batch and feature dimensions

# %%
np.random.seed(19)
BATCH_SIZE = 8
INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 32, 64, 32
rnn = RNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

print("W1:", rnn.W1.shape, "W2:", rnn.W2.shape, "W3:", rnn.W3.shape)

# %% [markdown]
# ### Question 2a — What do $W_1$, $W_2$, and $W_3$ map?
#
# | Matrix | Maps from | Maps to | Shape in this PCW |
# |:---|:---|:---|:---|
# | $W_1$ | \`input_dim\` | \`hidden_dim\` | $(32,64)$ |
# | $W_2$ | \`hidden_dim\` | \`hidden_dim\` | $(64,64)$ |
# | $W_3$ | \`hidden_dim\` | \`output_dim\` | $(64,32)$ |
#
# $W_2$ applies the **recurrent state transition**, not a transition between two independent feedforward layers. Biases $b_1$ and $b_2$ have lengths 64 and 32. The weights do not depend on batch size because the same matrices act on all eight samples.

# %% [markdown]
# ### Code Cell 3 — One timestep

# %%
X = np.random.uniform(size=(BATCH_SIZE, INPUT_DIM))
H = np.zeros(shape=(BATCH_SIZE, HIDDEN_DIM))
H, Y = rnn.step(X, H)

assert H.shape == (8, 64)
assert Y.shape == (8, 32)
print("Input X:", X.shape, "| Updated H:", H.shape, "| Output Y:", Y.shape)

# %% [markdown]
# ### Question 2b — Inputs and outputs of \`step(X, H)\`
#
# **Inputs:** $X_t\in\mathbb R^{8\times32}$ contains **current timestep** vectors for eight independent sequences; $H_{t-1}\in\mathbb R^{8\times64}$ is the **previous hidden state**. At the beginning of a sequence, $H_{-1}$ is initialized to zeros.
#
# **Outputs:** $H_t\in\mathbb R^{8\times64}$ is the updated hidden state, and $Y_t\in\mathbb R^{8\times32}$ is the current output. Each row is calculated separately; the model mixes features *within a sample* but never information *between samples*.

# %% [markdown]
# ### Code Cell 4 — Three timesteps, as in the PCW

# %%
XS = [np.random.uniform(size=(BATCH_SIZE, INPUT_DIM)) for _ in range(3)]
H = np.zeros(shape=(BATCH_SIZE, HIDDEN_DIM))

H, Y0 = rnn.step(XS[0], H)
H, Y1 = rnn.step(XS[1], H)
H, Y2 = rnn.step(XS[2], H)

print("Three output shapes:", Y0.shape, Y1.shape, Y2.shape)
print("Final hidden-state shape:", H.shape)

# %% [markdown]
# ### Question 2c — Why reuse $H$ for all three timesteps?
#
# We reuse \`H\` because it carries the **updated hidden representation of the same batch element** to the next timestep. The code performs **three** steps:
#
# $$
# \begin{aligned}
# H_0 &= \tanh(XS[0]W_1 + H_{-1}W_2+b_1),\\
# H_1 &= \tanh(XS[1]W_1 + H_0W_2+b_1),\\
# H_2 &= \tanh(XS[2]W_1 + H_1W_2+b_1).
# \end{aligned}
# $$
#
# Thus $Y_2$ may depend on $XS[0]$, $XS[1]$, and $XS[2]$. The expression \`XS[0][0], XS[1][0], XS[2][0]\` follows **one sequence** through time; \`XS[0][1]\` follows a **different, independent sequence**. The variable name is reused to overwrite its *value* with the newest hidden state, not to retrain or change the parameter matrices.

# %% [markdown]
# ### Sanity check — Recurrence versus batch independence
#
# A small deterministic check: modifying only sample 0's first timestep should alter sample 0's later prediction, but not sample 1's.

# %%
def run_three_steps(inputs, network):
    hidden = np.zeros((inputs[0].shape[0], HIDDEN_DIM))
    outputs = []
    for xt in inputs:
        hidden, yt = network.step(xt, hidden)
        outputs.append(yt)
    return hidden, outputs

_, baseline_outputs = run_three_steps(XS, rnn)
perturbed_XS = [x.copy() for x in XS]
perturbed_XS[0][0, :] += 0.5
_, changed_outputs = run_three_steps(perturbed_XS, rnn)

assert not np.allclose(baseline_outputs[2][0], changed_outputs[2][0])
assert np.allclose(baseline_outputs[2][1:], changed_outputs[2][1:])
print("Earlier input affects its own later output:", True)
print("Other batch elements stay independent:", True)

# %% [markdown]
# ## Question 5 of 7 — Shakespeare RNN extension
#
# The PCW asks for the official TensorFlow tutorial **through “Generate text.”** The architecture in this section is **character-level**, using a **GRU** rather than an LSTM or the NumPy vanilla RNN above.
#
# **Data pipeline:** Shakespeare text $\to$ Unicode characters $\to$ integer IDs $\to$ 101-character chunks $\to$ shifted 100-character input/target pairs $\to$ batches of 64.
#
# **Model:** an embedding lookup (256 features) $\to$ a GRU (1,024 hidden units) $\to$ a dense output layer containing a logit for each possible next character.
#
# **Environment:** TensorFlow's tutorial explicitly selects legacy \`tf.keras\` via \`TF_USE_LEGACY_KERAS=1\`; \`tf-keras\` is included in this repository's requirements. The variable must be set **before** importing TensorFlow. GPU acceleration is recommended for the default training run.

# %% [markdown]
# ### Code Cell 5 — Setup, download, and examine the Shakespeare corpus

# %%
import time
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)
with open(path_to_file, "rb") as f:
    source_text = f.read().decode("utf-8")
vocab = sorted(set(source_text))
print("Corpus characters:", len(source_text), "| Unique characters:", len(vocab))
print(repr(source_text[:150]))

# %% [markdown]
# ### Code Cell 6 — Convert characters to integer IDs and back

# %%
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

example_ids = ids_from_chars(tf.strings.unicode_split(["Hello"], "UTF-8"))
print("Character IDs:", example_ids)
print("Round trip:", text_from_ids(example_ids).numpy())

# %% [markdown]
# ### Code Cell 7 — Prepare next-character targets
#
# A 101-character chunk such as \`Hello\` (in a smaller example) is split into overlapping input and target streams: \`Hell\` predicts \`ello\`. **Targets are shifted by one character**, not by one word.

# %%
all_ids = ids_from_chars(tf.strings.unicode_split(source_text, "UTF-8"))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 100
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(sequence):
    return sequence[:-1], sequence[1:]

training_examples = sequences.map(split_input_target)
for first_x, first_y in training_examples.take(1):
    print("Input preview:", repr(text_from_ids(first_x).numpy().decode("utf-8")[:70]))
    print("Target preview:", repr(text_from_ids(first_y).numpy().decode("utf-8")[:70]))

# %% [markdown]
# ### Code Cell 8 — Shuffle and batch
#
# Each sample is a length-100 sequence, so tensors have shape $(64,100)$. Recurrence happens **within rows across 100 timesteps**; other rows are independent.

# %%
SHAKESPEARE_BATCH_SIZE = 64
BUFFER_SIZE = 10_000
dataset = (
    training_examples.shuffle(BUFFER_SIZE)
    .batch(SHAKESPEARE_BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
MAX_BATCHES = int(os.getenv("CS156_MAX_BATCHES", "0"))
if MAX_BATCHES > 0:
    dataset = dataset.take(MAX_BATCHES)
print("Training dataset:", dataset.element_spec)

# %% [markdown]
# ### Code Cell 9 — Embedding $\to$ GRU $\to$ Dense
#
# For an input batch $X$ with shape $(B,T)$:
#
# $$
# \begin{aligned}
# \operatorname{Embedding}(X)&\in\mathbb R^{B\times T\times256},\\
# \operatorname{GRU}(\cdot)&\in\mathbb R^{B\times T\times1024},\\
# \operatorname{Dense}(\cdot)&\in\mathbb R^{B\times T\times|V|}.
# \end{aligned}
# $$
#
# The GRU maintains a state of shape $(B,1024)$. The final Dense layer produces **raw logits**; softmax is handled implicitly in the cross-entropy loss or by categorical sampling.

# %%
vocab_size = len(ids_from_chars.get_vocabulary())
embedding_dim = int(os.getenv("CS156_EMBEDDING_DIM", "256"))
rnn_units = int(os.getenv("CS156_GRU_UNITS", "1024"))

class ShakespeareModel(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_size, units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocabulary_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        logits = self.dense(x, training=training)
        if return_state:
            return logits, states
        return logits

model = ShakespeareModel(vocab_size, embedding_dim, rnn_units)

# %% [markdown]
# ### Code Cell 10 — Inspect model outputs before training

# %%
for sample_x, sample_y in dataset.take(1):
    sample_logits = model(sample_x)
    print("Input:", sample_x.shape, "Targets:", sample_y.shape)
    print("Logits:", sample_logits.shape)
    assert sample_logits.shape == (
        SHAKESPEARE_BATCH_SIZE, seq_length, vocab_size
    )
model.summary()
sample_ids = tf.squeeze(tf.random.categorical(sample_logits[0], 1), axis=-1)
print("Untrained prediction:", repr(text_from_ids(sample_ids).numpy().decode("utf-8")[:120]))

# %% [markdown]
# ### Code Cell 11 — Train using BPTT and cross-entropy
#
# At each timestep, the correct label is the actual next character. \`SparseCategoricalCrossentropy(from_logits=True)\` aggregates these prediction errors; Adam updates all trainable embedding, GRU, and Dense weights. \`model.fit\` differentiates through the **unrolled GRU sequence**, which is backpropagation through time.
#
# Default hyperparameters match the tutorial. To run a small smoke test, set \`CS156_EPOCHS=1 CS156_GRU_UNITS=64 CS156_MAX_BATCHES=2 CS156_GENERATE=25\` in the environment **before starting** the notebook/kernel or script.

# %%
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
initial_loss = loss_function(sample_y, sample_logits)
print("Initial loss:", float(initial_loss.numpy()), "| exp(loss):", float(tf.exp(initial_loss)))
model.compile(optimizer="adam", loss=loss_function)

checkpoint_dir = os.getenv("CS156_CHECKPOINT_DIR", "./training_checkpoints/session19")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}"),
    save_weights_only=True,
)
EPOCHS = int(os.getenv("CS156_EPOCHS", "20"))
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# %% [markdown]
# ### Code Cell 12 — Visualize training loss
#
# The loss displayed here is a **training** metric, not held-out generalization performance. Output will vary across runs; the source screenshot did not include model training results.

# %%
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, EPOCHS + 1), history.history["loss"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Sparse categorical cross-entropy")
plt.title("Shakespeare GRU — training loss")
plt.grid(alpha=0.25)
if os.getenv("CS156_NO_SHOW", "0") == "1":
    plt.close()
else:
    plt.show()
print("Last training loss:", history.history["loss"][-1])

# %% [markdown]
# ### Code Cell 13 — Generate one character at a time
#
# After processing the prompt, use the **final** timestep's logits. Divide by a positive **temperature**, disallow the special \`[UNK]\` token, sample a character, and feed the sampled character **and the new GRU hidden state** into the next call. Repeat for an autoregressive output.

# %%
class OneStep(tf.keras.Model):
    def __init__(self, trained_model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = float(temperature)
        self.trained_model = trained_model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        unk_id = int(ids_from_chars(tf.constant(["[UNK]"])).numpy()[0])
        self.prediction_mask = tf.one_hot(
            unk_id, len(ids_from_chars.get_vocabulary()),
            on_value=-1e9, off_value=0.0
        )

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        logits, states = self.trained_model(
            inputs=input_ids, states=states, return_state=True
        )
        last_logits = logits[:, -1, :] / self.temperature
        next_ids = tf.random.categorical(
            last_logits + self.prediction_mask, num_samples=1
        )
        next_chars = self.chars_from_ids(tf.squeeze(next_ids, axis=-1))
        return next_chars, states

TEMPERATURE = float(os.getenv("CS156_TEMPERATURE", "1.0"))
generator = OneStep(model, chars_from_ids, ids_from_chars, TEMPERATURE)

def generate_text(prompts, count=1000):
    states = None
    current = tf.constant(prompts)
    generated = [current]
    for _ in range(count):
        current, states = generator.generate_one_step(current, states)
        generated.append(current)
    return [item.decode("utf-8") for item in tf.strings.join(generated).numpy()]

GENERATE_LENGTH = int(os.getenv("CS156_GENERATE", "1000"))
start = time.time()
for sample in generate_text(["ROMEO:"], GENERATE_LENGTH):
    print(sample)
print("Generation seconds:", round(time.time() - start, 2))

# %% [markdown]
# ### Code Cell 14 — Generate several independent sequences in a batch

# %%
start = time.time()
for index, generated_text in enumerate(
    generate_text(["ROMEO:"] * 5, GENERATE_LENGTH), start=1
):
    print("\n=== Sample", index, "===\n")
    print(generated_text)
print("Batched generation seconds:", round(time.time() - start, 2))

# %% [markdown]
# ## Question 6 of 7 — Shakespeare RNN: conceptual answers
#
# ### 6a. What does \`tf.keras.layers.Embedding\` do?
#
# An embedding layer is a **trainable lookup table** mapping each discrete character ID to a dense vector. Here, each ID maps to 256 real-valued features. The vectors are learned by gradient descent as part of next-character prediction; they are **not** handcrafted word definitions or pretrained semantic embeddings.
#
# ### 6b. What is a GRU, and why use one instead of a vanilla RNN?
#
# A **Gated Recurrent Unit** uses an **update gate** to balance remembered versus newly computed state, and a **reset gate** to control how much past state informs the candidate state. Its additive update path makes it easier for gradients to flow backward through longer sequences than the repeated nonlinear transformations in a vanilla RNN. This **mitigates vanishing gradients**, though neither vanishing nor exploding gradients are impossible. Unlike an LSTM, a standard GRU has no separate cell-state vector.
#
# ### 6c. How does text generation work?
#
# 1. Read Shakespeare's corpus, build a character vocabulary, and map each character to its integer ID.
# 2. Form length-100 input sequences with targets shifted forward by one character. Shuffle and batch examples.
# 3. Train **Embedding → GRU → Dense** to assign high logits to the true next character at every position, minimizing cross-entropy through BPTT.
# 4. Start from a prompt such as **ROMEO:**; turn prompt characters into IDs and process them through the network.
# 5. Extract logits from its last timestep; scale them by **temperature**, mask \`[UNK]\`, and randomly sample the next **character**.
# 6. Decode the sampled character to text; pass **both** it and the updated hidden state into the next generation call.
# 7. Repeat for the requested number of characters. Batch generation repeats this process for multiple **independent** sequences.
#
# **Key correction to the informal submitted PCW response:** The tutorial predicts **characters**, not entire words. The recurrent state helps later generated characters depend on prior context, but generated grammar and meaning are not guaranteed.
#
# ## Question 7 — Catch condition
#
# The PCW's final item is an **optional PDF upload** for notebook output, not an additional exercise. This repository includes executable code rather than pretending that a costly training run was already completed.
#
# ## References
#
# - [TensorFlow — Text generation with an RNN](https://www.tensorflow.org/text/tutorials/text_generation)
# - [StatQuest — Recurrent Neural Networks (RNNs) Clearly Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80&t=380s)
# - Supplied Session 19 PCW screenshot PDF (7 pages).
