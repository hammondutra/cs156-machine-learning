"""CS156 Session 17 — CNNs, filters, and deep learning.

Standalone version of the seven code cells in the Session 17 notebook.
Uses SciPy's sample raccoon photo (which may download on its first use).
Run with: python src/session17_cnns_filters_deep_learning.py
In VS Code, the # %% markers also support interactive cell execution.
"""

# %% [markdown]
# # CS156 — Session 17: CNNs, Filters, and Deep Learning
#
# **Source:** supplied 13-page *CS156 Session 17 PCW*, **CNNs, filters, and deep learning** (5 November 2026).
#
# This notebook is a polished, reproducible adaptation of **Questions 1–5**, preserving the **seven PCW code cells** in order. Question 6 is a file-upload submission prompt and has no separate computational exercise. The long response in Question 1 has been edited into readable Jupyter Markdown with rendered mathematical notation rather than pasted as unformatted rich text.
#
# **Learning goals:** convolution vs. cross-correlation; shared kernel weights; discrete image filters; pooling; feature maps; edge detection with Sobel operators; multichannel convolution; image-gradient magnitude; efficient convolution; and backpropagation through convolutional layers.
#
# **Dependencies:** NumPy, SciPy, Matplotlib, and a Python 3 Jupyter kernel (covered by the repository requirements). SciPy's modern **datasets.face** raccoon image may be **downloaded and cached on its first use**; an internet connection is needed the first time unless already cached. The older **misc.face** is no longer available in recent SciPy releases.
#
# **Run order:** execute top-to-bottom. The convolution and Sobel outputs are illustrative **fixed-filter** experiments, not learned CNN training results.

# %% [markdown]
# ## Question 1 of 6 — CNN introduction and LLM dialogue (edited for readability)
#
# ### The two original prompts
#
# **Prompt 1 — High-level explanation**
#
# > Give me a high-level explanation of a Convolutional Neural Network.
#
# **Prompt 2 — Detailed mechanism and applications**
#
# > Explain in detail the step-by-step process of a convolution (including the mathematics and computational methods apply to discretize/optimize/etc these process in computers), as well as how we might use convolutions to convert images.
#
# **Discussion and follow-ups included in the source PCW:** how pooling works; how multiple filters learn different patterns; how a kernel produces one output per spatial location; how an eye-template correlation map should be interpreted; and how Sobel filters reveal directional image gradients. The edited explanations and equations below organize those exchanges by concept.
#
# ### 1. CNNs: the broad picture
#
# A **convolutional neural network (CNN)** is a neural network whose architecture exploits the **spatial structure** of images. Instead of flattening the image immediately and connecting every feature to every neuron, it applies **local filters** to image patches.
#
# A typical classification pipeline is:
#
# 1. **Convolution:** apply multiple filters to produce **feature maps**.
# 2. **Nonlinear activation:** commonly a ReLU, $a(z)=\max(0,z)$.
# 3. **Optional pooling or strided convolution:** reduce spatial resolution.
# 4. **Deeper layers:** combine local patterns into larger structures.
# 5. **Prediction head:** combine features into class scores/probabilities.
#
# Early learned filters may react to edges, color contrasts, and textures. Later layers can combine these responses into more complex patterns relevant to the task. Such interpretations are illustrative; CNN filters are optimized for prediction, not explicitly labeled as "eye detector" or "ear detector."

# %% [markdown]
# ### 2. What pooling does
#
# Pooling **summarizes a small region** of a feature map. For a $2\times2$ max-pooling window with stride $2$,
#
# $$
# \begin{bmatrix}
# 1&3&2&4\\
# 5&6&1&2\\
# 7&2&8&3\\
# 4&1&5&9
# \end{bmatrix}
# \longrightarrow
# \begin{bmatrix}
# 6&4\\
# 7&9
# \end{bmatrix}.
# $$
#
# The number of spatial elements decreases from $16$ to $4$. **Max pooling** retains the largest activation in each region; **average pooling** retains the region's arithmetic mean.
#
# Pooling reduces subsequent computation and offers limited tolerance to small spatial shifts, but discards precise location information. Standard max/average pooling does not learn parameters.
#
# ### 3. How one kernel learns and how multiple kernels coexist
#
# A $3\times3$ grayscale kernel has nine learned weights:
#
# $$
# K=\begin{bmatrix}
# w_1&w_2&w_3\\
# w_4&w_5&w_6\\
# w_7&w_8&w_9
# \end{bmatrix}.
# $$
#
# One kernel slides across **all valid spatial positions** with the **same weights** and produces **one feature map**. A convolutional layer can learn many kernels simultaneously: 32 kernels produce 32 output feature maps (assuming each outputs one channel).
#
# During a forward pass the kernel weights remain fixed. Across training updates, backpropagation provides gradients and an optimizer changes the weights:
#
# $$
# K^{(t+1)}=K^{(t)}-\eta\nabla_K L.
# $$
#
# A deeper layer sees the prior layer's feature maps rather than necessarily seeing the original pixels. Repeating convolutions *across positions* is distinct from repeating *optimization steps over training batches*.

# %% [markdown]
# ### 4. From continuous convolution to discrete cross-correlation
#
# A continuous two-dimensional **convolution** is
#
# $$
# (I*K)(x,y)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
# I(x-u,y-v)\,K(u,v)\,du\,dv.
# $$
#
# The function $K$ is reversed with respect to both spatial axes. CNN frameworks usually implement **cross-correlation** instead: the kernel is not flipped.
#
# For a discrete image $X$ and a $k_h\times k_w$ filter $K$, a valid, unit-stride, single-channel cross-correlation is
#
# $$
# \boxed{Y[i,j]=\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}
# X[i+u,j+v]K[u,v].}
# $$
#
# An integral over a continuous region becomes a **finite sum** over pixel samples. If deriving a physical approximation to a continuous integral, pixel spacing factors $\Delta x\Delta y$ may also appear; standard CNN layers generally use the unscaled discrete sum.
#
# For every output position: extract a patch, multiply element-wise by the kernel, **sum the products**, and record one scalar. Thus **one kernel produces one output value at each spatial location**, rather than one number for the whole image.
#
# **Worked example:** take a vertical step edge and the Sobel kernel
#
# $$
# X=\begin{bmatrix}
# 0&0&0&10&10\\
# 0&0&0&10&10\\
# 0&0&0&10&10\\
# 0&0&0&10&10\\
# 0&0&0&10&10
# \end{bmatrix},
# \qquad
# K_x=\begin{bmatrix}
# 1&0&-1\\
# 2&0&-2\\
# 1&0&-1
# \end{bmatrix}.
# $$
#
# For the patch $P=\begin{bmatrix}0&0&10\\0&0&10\\0&0&10\end{bmatrix}$,
#
# $$
# \sum_{u,v}P[u,v]K_x[u,v]
# =-10-20-10=\boxed{-40}.
# $$
#
# The full **valid** correlation is
#
# $$
# Y=\begin{bmatrix}
# 0&-40&-40\\
# 0&-40&-40\\
# 0&-40&-40
# \end{bmatrix}.
# $$
#
# A negative result here means brighter values lie on the side carrying negative filter weights. **Edge strength** concerns the response magnitude, not only its sign.

# %% [markdown]
# ### 5. Stride, padding, dilation, and output size
#
# - **Stride $S$:** distance the kernel moves between output positions. Larger strides reduce output resolution and computation.
# - **Padding $P$:** values appended at image borders (zero, reflect, symmetric, etc.) to control boundary behavior and output dimensions.
# - **Dilation $D$:** spacing between kernel samples; it enlarges the effective receptive field without adding weights.
#
# For one spatial dimension,
#
# $$
# H_{\mathrm{out}}
# =\left\lfloor
# \frac{H+2P-D(K-1)-1}{S}+1
# \right\rfloor.
# $$
#
# The same formula applies to image width. With input $100\times100$, a $3\times3$ filter, stride $1$, dilation $1$, and no padding, the output is $98\times98$.
#
# ### 6. Different kernels, different image transformations
#
# A kernel's numerical weights define its operation:
#
# **Box blur (local mean):**
#
# $$
# K_{\mathrm{blur}}=\frac19\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}.
# $$
#
# **Gaussian-like blur (center-weighted smoothing):**
#
# $$
# K_{\mathrm{Gaussian}}=\frac1{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}.
# $$
#
# **Sharpening:**
#
# $$
# K_{\mathrm{sharp}}=\begin{bmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{bmatrix}.
# $$
#
# **Vertical-edge detection:**
#
# $$
# K_x=\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}.
# $$
#
# **Horizontal-edge detection:**
#
# $$
# K_y=\begin{bmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{bmatrix}.
# $$
#
# Blurring smooths rapid variations; sharpening amplifies local contrast; Sobel operators approximate directional intensity derivatives. A **single fixed convolution is linear**; complicated non-linear visual edits require additional operations or model layers.

# %% [markdown]
# ### 7. Why the Sobel filters approximate derivatives
#
# A central finite-difference derivative is
#
# $$
# f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.
# $$
#
# The Sobel kernels use **opposite-signed coefficients** on opposite sides of a pixel, approximating intensity changes. Their weights also smooth in the direction perpendicular to differentiation.
#
# With the signs used in this PCW,
#
# $$
# G_x\propto-\frac{\partial I}{\partial x},
# \qquad
# G_y\propto-\frac{\partial I}{\partial y}.
# $$
#
# The sign depends on kernel orientation; **gradient magnitude** is unaffected when both signs are reversed:
#
# $$
# \boxed{\|\nabla I\|\approx\sqrt{G_x^2+G_y^2}}.
# $$
#
# The gradient magnitude gives the *strength* of local changes independent of their direction. Its scalar output is not an arithmetic average of the two filtered images.
#
# ### 8. Multichannel convolution
#
# For a color image $X\in\mathbb{R}^{H\times W\times3}$, each full-depth RGB filter spans the three input channels:
#
# $$
# \boxed{Y[i,j]=
# \sum_{c=1}^{3}
# \sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}
# X[i+u,j+v,c]K[u,v,c]+b.}
# $$
#
# The $3\times3\times3=27$ products sum to **one output value for one filter at one position**. With $C_{\mathrm{out}}$ different filters, the output contains $C_{\mathrm{out}}$ channels.
#
# For example, with padding 1 and stride 1:
#
# $$
# 100\times100\times3\ \xrightarrow{\ 32\ \mathrm{filters}\ }\ 100\times100\times32.
# $$

# %% [markdown]
# ### 9. Computational implementation and optimizations
#
# The naive method nests loops over output positions, channels, and kernel coordinates:
#
# $$
# \text{work}\ \sim\
# O\!\left(
# H_{\mathrm{out}}W_{\mathrm{out}}C_{\mathrm{in}}
# C_{\mathrm{out}}k_hk_w
# \right).
# $$
#
# **Vectorization / im2col:** arrange image patches into rows of $X_{\mathrm{patches}}$ and flattened kernels into columns of $W$; then apply optimized matrix multiplication:
#
# $$
# Y_{\mathrm{flat}}=X_{\mathrm{patches}}W.
# $$
#
# This is fast on modern linear algebra hardware, although explicitly materializing the patches may cost substantial memory.
#
# **FFT convolution:** the convolution theorem gives
#
# $$
# \mathcal{F}(I*K)=\mathcal{F}(I)\mathcal{F}(K).
# $$
#
# FFT-based filtering can be faster for large kernels, but requires attention to padding and circular-convolution boundary effects. Code Cell 3 below uses **symmetric padding and FFT convolution with the flipped template** to reproduce the original PCW's same-size cross-correlation (up to floating-point roundoff), without the expensive direct calculation over a $65\times80$ template.
#
# **Separable filters:** if $K=ab^\top$, a $k\times k$ filter can be applied as successive one-dimensional filters; work falls from approximately $k^2$ to $2k$ multiplications per output location. GPU parallelism, reduced precision, Winograd algorithms, and depthwise-separable filters provide further practical speedups.
#
# ### 10. Backpropagation through shared weights
#
# For one grayscale cross-correlation layer,
#
# $$
# Z[i,j]=\sum_{u,v}X[i+u,j+v]K[u,v],
# $$
#
# the chain rule gives
#
# $$
# \boxed{
# \frac{\partial L}{\partial K[u,v]}=
# \sum_{i,j}
# \frac{\partial L}{\partial Z[i,j]}\,X[i+u,j+v].
# }
# $$
#
# The same kernel weight is used at **many positions**. Hence its gradient accumulates the contributions from all those uses, and gradient descent can update it efficiently.

# %% [markdown]
# ### 11. Follow-up to Question 1 — Tied weights (under 150 words)
#
# **Tied weights** means that the same kernel coefficients are reused at every spatial location. Instead of learning a separate detector for each pixel position, a convolutional layer learns a small set of filters and applies each one across the image. This reduces parameter count and makes local pattern detection approximately translation-equivariant, aside from boundary and sampling effects. Deep CNN predictions depend on these shared coefficients through many convolutional layers and nonlinear activations. The **chain rule** is essential because backpropagation must determine how a change in each shared weight affects the final loss through all intermediate computations. Since a weight participates at many image positions, its gradient **sums contributions from all of them**. An optimizer uses these accumulated gradients to update the kernel.
#
# ### 12. Suggested improvements to the LLM dialogue
#
# The first prompt was useful for intuition; the second made the mathematics and implementation explicit. A stronger study prompt would request **one worked numerical patch calculation**, distinguish **convolution vs. cross-correlation**, specify **padding/stride and output dimensions**, and require **runnable code with the actual PCW image**. The explanations above also correct the misleading suggestion that combining Sobel derivatives by Euclidean magnitude is "averaging" them.

# %% [markdown]
# ## Question 2 of 6 — Core question: convolutions
#
# The PCW treats one filter at a time using SciPy's sample raccoon image. The eye patch acts as a **template** for cross-correlation. Later, a predefined Sobel operator detects intensity changes.
#
# ### Code Cell 1 of 7 — Load and center the raccoon image
#
# The source used **scipy.misc.face(gray=True)**, which is removed in modern SciPy. We use **scipy.datasets.face(gray=True)**, which has the same 768×1024 sample image. Centering the image changes its numerical intensities but leaves the spatial patterns intact.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

try:
    from scipy.datasets import face as load_raccoon
except ImportError:  # Compatibility with older SciPy
    from scipy.misc import face as load_raccoon

try:
    original_face = load_raccoon(gray=True)
except Exception as exc:
    raise RuntimeError(
        "Could not load the SciPy raccoon image. Modern scipy.datasets.face "
        "downloads it on first use; connect to the internet and rerun, "
        "or use an environment where the image is already cached."
    ) from exc

face = original_face.astype(np.float64)
face -= face.mean()

print("Image shape:", face.shape, "| dtype:", face.dtype)
plt.figure(figsize=(10, 6))
plt.imshow(face, cmap="Greys")
plt.colorbar(label="Mean-centered intensity")
plt.title("SciPy raccoon (grayscale, mean-centered)")
plt.show()

# %% [markdown]
# ### Code Cell 2 of 7 — Extract an eye template
#
# The PCW takes the rectangle at rows **300:365** and columns **670:750**, giving a $65\times80$ kernel. Subtract the template's own mean so its weights summarize local departures from average intensity.

# %%
template = face[300:365, 670:750].copy()
template -= template.mean()

print("Template shape:", template.shape)
plt.figure(figsize=(7, 4))
plt.imshow(template, cmap="Greys")
plt.colorbar(label="Centered intensity")
plt.title("Eye template (65 × 80)")
plt.show()

# %% [markdown]
# ### Code Cell 3 of 7 — Correlate the eye template with the whole image
#
# The original PCW uses **signal.correlate2d(face, template, boundary='symm', mode='same')**. Because the template has $5{,}200$ weights, a direct correlation performs billions of multiplies. The following helper uses an FFT for speed while retaining **symmetric boundary extension** and the same output alignment.
#
# **Interpretation:** each output pixel is the dot product between the template and a same-sized local patch. The image is a **cross-correlation / response map**, not a reconstructed raccoon. Its high values indicate strong unnormalized template responses. The **other eye** and **nose** can also activate because they share light/dark transitions and local textures with the template. Correlation does not know the semantic category "eye", and response magnitude is affected by contrast and signal energy.

# %%
def correlate_same_symmetric_fft(image, kernel):
    """Equivalent to correlate2d(..., boundary='symm', mode='same')."""
    pads = tuple(((k - 1) // 2, k // 2) for k in kernel.shape)
    extended = np.pad(image, pads, mode="symmetric")
    # FFT convolution with a flipped kernel implements cross-correlation.
    return signal.fftconvolve(extended, kernel[::-1, ::-1], mode="valid")

corr_template = correlate_same_symmetric_fft(face, template)

plt.figure(figsize=(10, 6))
plt.imshow(corr_template, cmap="Greys")
plt.colorbar(label="Unnormalized correlation")
plt.title("Raccoon eye-template correlation map")
plt.show()

# %% [markdown]
# ### Code Cell 4 of 7 — Inspect the vertical-edge Sobel kernel
#
# The first Sobel kernel has opposite signs **across columns**:
#
# $$
# K_x=\begin{bmatrix}
# 1&0&-1\\
# 2&0&-2\\
# 1&0&-1
# \end{bmatrix}.
# $$
#
# It responds to **left-right** changes in brightness, which form **vertical** edges. The signed response reverses when the brightness transition reverses.

# %%
sobel_x = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=float)

plt.figure(figsize=(5, 4))
plt.imshow(sobel_x, cmap="Greys", vmin=-2, vmax=2)
plt.colorbar(label="Kernel weight")
plt.title("Sobel x: vertical-edge detector")
plt.show()

# %% [markdown]
# ### Code Cell 5 of 7 — Apply vertical-edge detection
#
# A single Sobel-filtered image shows **signed directional changes**. Edge locations with strong positive or negative responses stand out; near-uniform areas remain close to zero.

# %%
vertical_edges = signal.correlate2d(
    face, sobel_x, boundary="symm", mode="same"
)

plt.figure(figsize=(10, 6))
plt.imshow(vertical_edges, cmap="Greys")
plt.colorbar(label="Signed x-direction response")
plt.title("Raccoon: vertical edges (Sobel x)")
plt.show()

# %% [markdown]
# ## Question 3 of 6 — Questions 2b and 2c
#
# **Q2b — Horizontal or vertical detector?** The first Sobel operator is a **vertical-edge detector**: its positive left column and negative right column compare brightness *horizontally*. Therefore, it responds strongly at boundaries that run vertically.
#
# **Q2c — How does a filter operate on RGB images?** A standard RGB filter has shape $k_h\times k_w\times3$. At every position, multiply corresponding pixel and kernel entries across **all three channels**, sum those products, and optionally add a bias. The result is **one scalar per filter per spatial location**, not three independent output values. A bank of $C_{\mathrm{out}}$ filters produces $C_{\mathrm{out}}$ output channels.
#
# $$
# Y[i,j,o]=\sum_{c=1}^{C_{\mathrm{in}}}
# \sum_{u,v}X[i+u,j+v,c]K[u,v,c,o]+b_o.
# $$

# %% [markdown]
# ## Question 4 of 6 — Extension: the other Sobel operator
#
# The second Sobel filter compares intensity **above versus below** a pixel and therefore emphasizes **horizontal** edges. Compared with the vertical-edge response, it highlights different contours, such as approximately horizontal boundaries around the eyes and mouth.

# %%
sobel_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
], dtype=float)

horizontal_edges = signal.correlate2d(
    face, sobel_y, boundary="symm", mode="same"
)

plt.figure(figsize=(10, 6))
plt.imshow(horizontal_edges, cmap="Greys")
plt.colorbar(label="Signed y-direction response")
plt.title("Raccoon: horizontal edges (Sobel y)")
plt.show()

# %% [markdown]
# ## Question 5 of 6 — Combine both filters into the gradient magnitude
#
# The two filters approximate directional derivatives. Combining them with the **Euclidean norm** produces a nonnegative edge-strength map:
#
# $$
# \boxed{G[i,j]=\sqrt{G_x[i,j]^2+G_y[i,j]^2}.}
# $$
#
# Compared with either directional filter alone, this highlights **vertical, horizontal, and diagonal boundaries** together. The result is **not an average**: a strong response in *either* direction increases the magnitude. If both components have magnitude $a$, the combined magnitude is $\sqrt{2}a$. Uniform regions remain near zero.
#
# In the PCW screenshot, the combined gradient emphasizes contours around the raccoon's face, ears, nose, and surrounding plant leaves. The bright/dark appearance depends on the chosen grayscale colormap (here **Greys** uses darker tones for higher values).

# %%
# The two directional responses come from Code Cells 5 and 6.
gradient = np.sqrt(horizontal_edges**2 + vertical_edges**2)

plt.figure(figsize=(10, 7))
plt.imshow(gradient, cmap="Greys")
plt.colorbar(label="Gradient magnitude")
plt.title("Combined Sobel gradient magnitude")
plt.show()

# %% [markdown]
# ## Question 6 of 6 — Submission note
#
# The supplied PCW ends with an **optional file-upload response** if the student chooses to compile the full PCW into a code notebook. This repository notebook serves as that editable computational companion; no additional uploaded response or completion claim is invented.
#
# ## Key takeaways
#
# 1. A fixed kernel **filters** a local neighborhood; a CNN **learns** many such filters.
# 2. Ordinary CNN "convolution" is normally **cross-correlation** (no filter flip).
# 3. A kernel generates **one scalar per position**, and a bank of kernels generates a stack of feature maps.
# 4. Sobel operators approximate directional **image derivatives**; the Euclidean magnitude combines edge strengths in both directions.
# 5. Weight sharing reduces parameters and makes gradients accumulate across every location where a filter is used.
# 6. Efficient implementations use compiled loops, **matrix multiplication**, **FFT methods**, or **separable filters**, depending on kernel size and hardware.
#
# **Data provenance:** raccoon image from the SciPy sample dataset; questions and seven original exercises from the supplied CS156 Session 17 PCW. Math, explanation formatting, compatibility handling, and FFT acceleration are editorial improvements to the repository version.

