# Brawl Stars — CS156 First Pipeline

A machine learning pipeline that predicts the winning team in **3v3 Brawl Stars** matches from the six selected brawlers. Logistic regression, feature encoding, data splitting, and evaluation are implemented from first principles in Python.

## Dataset

- **162 real matches** across six game modes, including **93 Brawl Ball matches**.
- **107 distinct brawlers** represented in the collected compositions.
- **324 records** including side-swapped versions used in augmentation experiments.
- [Original dataset in Google Sheets](https://docs.google.com/spreadsheets/d/1PT7GDSb56HKSjGbvmj5-bGVWCmbP8Wo7-tSkRxEk7aQ/edit)

## What the pipeline does

1. Loads and validates the match records and paired side-swapped observations.
2. Encodes team compositions as signed or separate-team numeric features.
3. Performs **five-fold stratified cross-validation**, augmenting training folds only.
4. Trains regularized **binary logistic regression from scratch** using gradient descent.
5. Compares **six conditions (E0–E5)**: prior baseline, feature encodings, augmentation, and Brawl Ball specialization.
6. Evaluates accuracy, balanced accuracy, log loss, Brier score, confusion matrices, and training progress.

## Results

| Evaluation | Best model | Accuracy | Log loss |
| --- | --- | ---: | ---: |
| All 162 matches | E0 — historical-win-rate baseline | 74.7% | 0.566 |
| 93 Brawl Ball matches | E1 — all-mode signed-composition regression | 69.9% | 0.574 |

See the report for the full six-condition comparison and interpretation.

## Files and report

- [`data/`](data/) — original matches, augmented matches, and brawler vocabulary
- [`src/data_setup.py`](src/data_setup.py) — data preparation and cross-validation
- [`src/regression_implementation.ipynb`](src/regression_implementation.ipynb) — from-scratch model training and metrics
- [`src/visualize_results.ipynb`](src/visualize_results.ipynb) — model-performance visualizations
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — project development history
- **[Final report (PDF)](report/first_pipeline_report.pdf)** · [Editable Google Doc](https://docs.google.com/document/d/1V4xB0lhuQDPl5ivvW_u_FC1eHQO9p__FKWfzDlCij3Y/edit)

To run the notebooks locally, create a Python environment, install `ipykernel` and `matplotlib`, and open the notebooks in VS Code or Jupyter.
