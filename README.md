# CS156 - Finding Patterns in Data with Machine Learning

A learning repository for machine-learning concepts, mathematical foundations, experiments, and reusable implementations studied in CS156 at Minerva University.

## Structure

- `notebooks/` - concept explanations, derivations, experiments, visualizations, and reflections.
- `src/` - reusable Python implementations extracted from the notebooks.

## Current topics

### Session 1 — Introduction to Machine Learning

- representation, probability, optimization, and generalization,
- Iris tabular data,
- MNIST image vectors,
- probability and Bayes' rule.

Open [`notebooks/01_introduction_to_machine_learning.ipynb`](notebooks/01_introduction_to_machine_learning.ipynb).

### Session 2 — Linear and Logistic Regression

- linear regression,
- logistic regression,
- loss functions,
- sensitivity / recall,
- implementing regression models from scratch.

Open [`notebooks/02_linear_and_logistic_regression.ipynb`](notebooks/02_linear_and_logistic_regression.ipynb).

### Session 3 — Multivariate Linear Regression

- matrix representation of regression,
- Titanic fare prediction,
- ordinary least squares,
- normal equations,
- residual geometry and SSR.

Open [`notebooks/03_multivariate_linear_regression.ipynb`](notebooks/03_multivariate_linear_regression.ipynb).

### Session 4 — Decision Trees

The existing Session 4 notebook is stored at:

[`notebooks/04_decision_trees.ipynb`](notebooks/04_decision_trees.ipynb)

### Session 5 — Naive Bayes

- joint, marginal, and conditional probability,
- conditional independence,
- TF-IDF,
- Multinomial Naive Bayes,
- learned parameters,
- linear decision boundaries,
- confusion matrices,
- feature probabilities and Laplace smoothing.

Open [`notebooks/05_naive_bayes_sms.ipynb`](notebooks/05_naive_bayes_sms.ipynb).

### Session 6 — Logistic Regression and Maximum Likelihood Estimation

- Gaussian-MLE preclass review,
- logistic-regression parameters and Bernoulli likelihood,
- log-likelihood of synthetic *Mujhe Chahiye Bollywood* data,
- comparing proposed parameter values and deriving the log-likelihood gradient,
- optional numerical MLE and 95% probability threshold.

Open [`notebooks/06_maximum_likelihood_estimation.ipynb`](notebooks/06_maximum_likelihood_estimation.ipynb).

### Session 7 — Feed-Forward Neural Networks and Perceptrons

- MNIST MLP (784 → 128 → 64 → 10): matrix dimensions, training, validation, test metrics and confusion matrix,
- single-layer Sonar perceptron trained from scratch with stratified 3-fold cross-validation,
- Fashion-MNIST binary top-versus-trouser experiment with optional PCA and learning curves.

Open [`notebooks/07_feed_forward_nn.ipynb`](notebooks/07_feed_forward_nn.ipynb).

### Session 8 — Gradients: Multivariate Derivatives

- sigmoid derivatives and chain rule,
- manual scalar backpropagation through a computation graph,
- gradients of the dot product `w^T x`,
- vectorized backward passes,
- automatic differentiation with JAX,
- gradient-descent updates on model weights.

Open [`notebooks/08_gradients_multivariate_derivatives.ipynb`](notebooks/08_gradients_multivariate_derivatives.ipynb).

Reusable script: [`src/session08_gradients.py`](src/session08_gradients.py).

### Session 9 — Metrics and Cross-Validation

- stratified 80/20 train/test splits,
- logistic regression versus nonlinear SVMs on concentric-circle data,
- classifier probability surfaces,
- ROC curves, threshold tradeoffs, and ROC-AUC,
- train/validation/test separation,
- grid search over SVM `gamma` and `C`,
- final evaluation on an untouched test set.

Open [`notebooks/09_metrics_cross_validation.ipynb`](notebooks/09_metrics_cross_validation.ipynb).

Reusable script: [`src/session09_metrics_cross_validation.py`](src/session09_metrics_cross_validation.py).

### Session 13 — Support Vector Classifiers and the Kernel Trick

- nullspace, representer theorem, and SVC soft-margin/dual equations;
- linear, homogeneous quadratic, and RBF kernels and feature maps;
- linear and RBF SVC functions from the first PCW question;
- kernel comparisons on two-moons data;
- optional financial transaction classification (CSV not included);
- Yosemite Village NOAA 2011–2016 month prediction, with recorded CV/test results;
- supplementary SVR versus linear/RBF-feature regression.

Open [Session 13 notebook](notebooks/13_support_vector_classifiers_kernel_trick.ipynb).

Reusable script: [Session 13 Python module](src/session13_svm_kernel_trick.py).

### Session 15 — Basis Functions 2: Ridge and Lasso

- ill-posed parameter estimation, overfitting, and regularization;
- Ridge (L2) versus Lasso (L1), coefficient shrinkage and feature selection;
- synthetic logistic regression and mean negative log-likelihood from Session 6;
- loss-surface contour plots for NLL, squared L2 norm, and their weighted sum;
- shifting the minimizer using an L2 penalty and changing the regularization strength;
- repeated sampling (50 datasets), unregularized logistic regression vs. RidgeClassifier, coefficient variance, and the bias–variance tradeoff;
- compatibility notes for JAX and scikit-learn; comparisons across different classifier objectives are qualified.

Open [Session 15 notebook](notebooks/15_basis_functions_2_ridge_lasso.ipynb).

Reusable script: [Session 15 Python module](src/session15_ridge_lasso.py).

**Source:** [Minerva Session 15 — Basis Functions 2: Ridge and Lasso](https://forum.minerva.edu/app/courses/4077/sections/13535/classes/105085) and the supplied PCW PDF. The experiments are synthetic/offline; install dependencies from `requirements.txt`.

**Yosemite extension — Fourier basis + Ridge temperature forecasting**

- load the professor's raw whitespace-delimited Yosemite temperature Gist;
- model annual/daily sine and cosine cycles with Fourier harmonics and season×hour interactions;
- compare ordinary least squares with Ridge L2 regularization;
- construct past 24/48/72/168-hour temperature lag features for rolling 24-hour-ahead forecasting;
- tune model settings on 2015 validation, with 2016 held out for final regression RMSE/MAE/R² evaluation;
- include an optional histogram gradient-boosting benchmark for higher predictive accuracy.

[Run the Yosemite notebook](notebooks/15_yosemite_fourier_ridge.ipynb) · [Python version](src/session15_yosemite_fourier_ridge.py).

The original Bollywood PCW notebook remains unchanged. This extension requires internet access to download the Gist data. No new Yosemite test scores are reported until the notebook runs.

### Session 17 — CNNs, Filters, and Deep Learning

- convolution vs. cross-correlation, feature maps, tied/shared weights, ReLU, and pooling;
- discrete 2D kernels, stride, padding, dilation, multichannel RGB filtering, and output shapes;
- fixed filters for blurring, sharpening, and vertical/horizontal Sobel edge detection;
- SciPy raccoon **eye-template matching**, response maps, and why other facial features also activate;
- combining Sobel responses into the **Euclidean image-gradient magnitude**;
- kernel-gradient backpropagation and efficient implementations (im2col, FFT, separable filters);
- carefully typeset LaTeX derivations and polished answers to the first PCW question.

Open [Session 17 notebook](notebooks/17_cnns_filters_deep_learning.ipynb).

Standalone Python script: [Session 17 — runnable convolution exercises](src/session17_cnns_filters_deep_learning.py).

**Source:** supplied *CS156 Session 17 PCW — CNNs, filters, and deep learning* (13 pages; seven code cells). The notebook updates legacy \`scipy.misc.face\` to \`scipy.datasets.face\` and accelerates large-template correlation using an equivalent FFT computation with symmetric padding. The raccoon image may require internet access on first run; SciPy caches it afterward.

### Session 18 — Autoencoders and Nonlinear Dimensionality Reduction

- encoder–decoder networks, self-supervised representation learning, and reconstruction objectives;
- compressed **2D bottleneck** representations of 8×8 handwritten digits from scikit-learn;
- nonlinear dimensionality reduction versus PCA, and limitations of reconstruction-based features;
- JAX automatic differentiation and Adam optimization of pixelwise binary cross-entropy;
- evaluating reconstruction quality and visualizing a **2D latent-space scatterplot** colored by digit label;
- original **five-layer** network versus **seven-layer** encoder–decoder extension, with matched training settings and loss curves;
- optional autoregressive **teacher-forcing** experiment, distinguishing teacher-forced from free-running reconstruction.

Open [Session 18 notebook](notebooks/18_autoencoders.ipynb).

Standalone Python script: [Session 18 — runnable autoencoder exercises](src/session18_autoencoders.py).

**Source:** supplied *CS156 Session 18 PCW — Autoencoders* (10 pages; 13 code cells). The PDF reports reconstruction losses of **0.374349 (5 layers)** and **0.374409 (7 layers)** for the controlled depth comparison. The teacher-forcing extension reports **0.337815 teacher-forced** versus **0.442996 free-running** loss. These are source-reported training scores, not held-out test metrics or guaranteed scores for each rerun. The notebook and script reproduce the exercises using local scikit-learn digits data.

**Dependencies:** Session 18 additionally uses `optax` (included in `requirements.txt`). Training uses 3,000 + 3,000 + 1,000 update steps by default; for a short functionality check, set `CS156_STEPS=100 CS156_TEACHER_STEPS=50 CS156_NO_SHOW=1` before running the standalone script.


### Session 19 — Recurrent Neural Networks (RNNs), LSTMs, and GRUs

- sequential data and recurrence: hidden states, weight sharing, and batch independence;
- batched **NumPy vanilla RNN** with three learned transformations and explicit input/state/output shapes;
- three-timestep state propagation and a sanity check that information passes **through time within each sequence**, not between batch elements;
- LSTM forget/input/output gates, additive cell-state updates, and backpropagation through time (BPTT);
- vanishing versus exploding gradients, and how GRU gating helps preserve longer-term information;
- Shakespeare **character-level text generation** using TensorFlow: integer tokenization, 256-dimensional embeddings, 1,024-unit GRU, 100-character training windows, next-character cross-entropy, Adam, checkpoints, and sampled autoregressive generation;
- cleaned-up, typeset LaTeX answers and conceptual responses to PCW Questions 1–6.

Open [Session 19 notebook](notebooks/19_recurrent_neural_networks.ipynb).

Standalone Python script: [Session 19 — RNN and Shakespeare GRU exercises](src/session19_recurrent_neural_networks.py).

**Source:** supplied *CS156 Session 19 PCW — RNNs* (7 pages; four NumPy code cells) and the [official TensorFlow text-generation tutorial](https://www.tensorflow.org/text/tutorials/text_generation). The extension follows the tutorial through **Generate text**. Dataset download requires internet access. GPU training recommended; outputs are intentionally left unexecuted, not fabricated.

**Run:** \`pip install -r requirements.txt\` then \`python src/session19_recurrent_neural_networks.py\`. The default is 20 epochs and 1,000 generated characters; to smoke-test with a small model and dataset, run \`CS156_EPOCHS=1 CS156_GRU_UNITS=64 CS156_MAX_BATCHES=2 CS156_GENERATE=25 CS156_NO_SHOW=1 python src/session19_recurrent_neural_networks.py\`. The script enables TensorFlow's legacy \`tf.keras\` to match the tutorial, using the \`tf-keras\` package.

### Session 20 — Transfer Learning, Fine-tuning, and Visualizing Embeddings

- transfer learning from ImageNet-pretrained **MobileNetV2** to binary cat/dog classification;
- data loading, augmentation, mobile-network input preprocessing, and efficient `tf.data` pipelines;
- **feature extraction** by freezing the pretrained convolutional backbone and training a new binary classification head;
- **fine-tuning** by unfreezing selected upper layers and continuing gradient-based optimization with a smaller learning rate;
- comparing training/validation curves and evaluating a held-out test split;
- **PCW extension:** extracting 1,280-dimensional GlobalAveragePooling2D vectors using a separate **unfine-tuned** MobileNetV2, reducing them with **PCA to 2D**, and coloring points by cat/dog;
- evaluating an optional **linear logistic-regression probe** in 2D PCA space, with the caveat that PCA visualization does not establish full-dimensional linear separability;
- diagnosing a Colab dataset-extraction/path error and providing a portable data-directory fallback.

**Primary learning notebook:** [Session 20 — Transfer Learning and Fine-tuning](notebooks/20_transfer_learning_fine_tuning.ipynb) (runnable tutorial and PCA extension).

**Secondary PCW:** [Session 20 — Written PCW notes](notebooks/20_transfer_learning_pcw_notes.ipynb).

**Standalone Python:** [Session 20 — Transfer learning script](src/session20_transfer_learning.py).

**Source:** supplied *CS156 Session 20 PCW* and the [official TensorFlow transfer-learning tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning). The student's executed notebook reported strong 2D cat/dog separation; the repository notebook intentionally omits saved output blobs for a smaller Git diff, so run it for reproducible plots. First run downloads the dataset and ImageNet weights; GPU recommended. Run with `python src/session20_transfer_learning.py`, or shorten training with `CS156_EPOCHS=1 CS156_FINE_EPOCHS=1 CS156_NO_SHOW=1 python src/session20_transfer_learning.py`.

### Session 21 — Assignment 2 Prep: SetFit, Contrastive Learning, and Data Ethics Extension

This session prepares for Pipeline Assignment 2 by selecting a **method extending Unit 2** and implementing a small, runnable precursor. The selected method is **SetFit (Sentence Transformer Fine-Tuning)**: a few-shot technique that fine-tunes pretrained sentence representations using labeled text pairs before fitting a classification head. The planned application is a context-aware classifier for authorized versus potentially unsafe **AI-agent tool actions**. SetFit is **supervised classification**, not automatically an anomaly detector.

To keep the original PCW runnable in its 384 MB kernel, the supplied experiment **does not actually fine-tune a pretrained transformer**. Instead, it encodes 16 synthetic agent task/action descriptions as normalized bag-of-words vectors and optimizes a 12-dimensional linear embedding using a **pairwise cosine-similarity mean squared loss and a manually implemented gradient**. Across 120 distinct training pairs, the submitted run reported training loss **0.326 → 0.000**, average same-label cosine similarity **0.569 → 0.978**, and different-label similarity **0.616 → −0.005**. These are **training-pair results only**, not evidence of held-out detection accuracy, practical security effectiveness, or a working SetFit implementation.

**Session structure:**

- **Main PCW (Questions 1–4):** [Contrastive-learning preparation notebook](notebooks/21_assignment2_prep_contrastive.ipynb) — selected model, training data, learning method, metrics, citations, mathematical objective, NumPy experiment, and interpretation. The originally submitted two-slide presentation is summarized in this notebook.
- **Presentation companion (Questions 1–2):** [Two-slide SetFit overview and speaker notes](notebooks/21_setfit_presentation_companion.md) — diffable summary of the submitted two-slide presentation, without bundling a binary PDF.
- **Revisited Session 11 extension (Question 5):** [Ames housing neighborhood analysis notebook](notebooks/21_session11_extension_ames.ipynb) — historical housing prices by 25 neighborhoods, sale counts, medians, interquartile ranges, figure, and discussion of sampling and interpretation limitations.
- **Standalone Python:** [Session 21 combined script](src/session21_assignment2_prep.py) — runs both studies sequentially; avoids duplicating long blocks of notebook code.
- **Offline housing data:** [Verified Ames aggregate CSV](data/session21_ames_neighborhood_summary.csv) — 25 neighborhood summaries derived from **1,460** observations, *not* a replacement for record-level analysis.

**Run from repository root:**

```bash
python src/session21_assignment2_prep.py
python src/session21_assignment2_prep.py --from-raw --no-show
python src/session21_assignment2_prep.py --no-show --save-plot ames.png
```

The default script works offline with `numpy`, `pandas`, and `matplotlib`. The `--from-raw` switch downloads the public [Ames Housing training CSV](https://github.com/knaat/Ames-housing/blob/main/train.csv), cleans invalid/missing price and neighborhood records, and recomputes group summaries. Both Jupyter notebooks include saved explanatory content; the Ames notebook also includes a saved visualization prepared from the same validated aggregates. Rerun their cells to reproduce results.

**Further work for Pipeline 2:** load an actual pretrained Sentence Transformer, perform SetFit fine-tuning on adequately labeled and sanitized action data, create a task/scenario-disjoint test split, compare against frozen embeddings and other baselines, and report precision/recall, F1, PR-AUC, false-positive rates, and latency. No such model performance is claimed at this stage.

**References:** [Hugging Face SetFit](https://huggingface.co/docs/setfit/index); [SetFit SST-2 dataset](https://huggingface.co/datasets/SetFit/sst2); [AgentDojo](https://github.com/ethz-spylab/agentdojo); De Cock (2011) / [Ames Housing CSV](https://github.com/knaat/Ames-housing/blob/main/train.csv). Source: submitted *CS156 Session 21 PCW*, 19 November 2026.


**Gap:** Sessions 10–12 and 14 are intentionally deferred. Financial and supplementary SVR scores are not reported because they were not run in the source PCW.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then select the `.venv` Python interpreter / kernel in VS Code or Jupyter.

## Notes

Some notebooks download public datasets when first executed, so an internet connection is required for those cells. Session 6 uses synthetic data and runs offline. Session 7's MNIST and Fashion-MNIST data are cached by Keras; its Sonar CSV can be read locally or downloaded to `~/.cache/cs156`. Session 8 uses NumPy for the manual-gradient exercises and JAX for the automatic-differentiation preview. Session 9 runs fully offline because its concentric-circle dataset is generated with scikit-learn. Session 13's synthetic datasets also run offline. Session 15 generates synthetic Bernoulli datasets locally; its JAX plot helpers and 50-fit scikit-learn experiment are executable offline. Session 17's sample raccoon image uses SciPy's cached dataset (first use may download). Session 18's scikit-learn digits data runs fully offline; Session 19's NumPy RNN runs offline, while its Shakespeare GRU downloads the text dataset and benefits from GPU acceleration; the 2D bottleneck, deeper comparison, and recurrent teacher-forcing experiments use JAX + Optax. Financial transactions require an external CSV (`CS156_TRANSACTIONS_CSV` and optional `CS156_FINANCIAL_TARGET`); NOAA Yosemite downloads and the expensive models are opt-in with `CS156_RUN_YOSEMITE=1`.
