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

The existing Session 4 notebook can be kept as:

`notebooks/04_decision_trees.ipynb`

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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then select the `.venv` Python interpreter / kernel in VS Code or Jupyter.

## Notes

Some notebooks download public datasets when first executed, so an internet connection is required for those cells. Session 6 uses synthetic data and runs offline. Session 7's MNIST and Fashion-MNIST data are cached by Keras; its Sonar CSV can be read locally or downloaded to `~/.cache/cs156`.
