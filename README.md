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

**Gap:** Sessions 10–12 are intentionally deferred. Financial and supplementary SVR scores are not reported because they were not run in the source PCW.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then select the `.venv` Python interpreter / kernel in VS Code or Jupyter.

## Notes

Some notebooks download public datasets when first executed, so an internet connection is required for those cells. Session 6 uses synthetic data and runs offline. Session 7's MNIST and Fashion-MNIST data are cached by Keras; its Sonar CSV can be read locally or downloaded to `~/.cache/cs156`. Session 8 uses NumPy for the manual-gradient exercises and JAX for the automatic-differentiation preview. Session 9 runs fully offline because its concentric-circle dataset is generated with scikit-learn. Session 13's synthetic datasets also run offline. Financial transactions require an external CSV (`CS156_TRANSACTIONS_CSV` and optional `CS156_FINANCIAL_TARGET`); NOAA Yosemite downloads and the expensive models are opt-in with `CS156_RUN_YOSEMITE=1`.
