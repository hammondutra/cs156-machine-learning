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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then select the `.venv` Python interpreter / kernel in VS Code or Jupyter.

## Notes

Some notebooks download public datasets when first executed, so an internet connection is required for those cells.
