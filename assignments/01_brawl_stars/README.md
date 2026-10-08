# Brawl Stars — Pipeline Assignment 1 (CS156)

**Question:** Do six-character team compositions predict Brawl Ball match wins better than a majority-class baseline or a single-brawler proxy?

Data: 25 personal battle-history matches from two screenshots, transcribed to [Google Sheets](https://docs.google.com/spreadsheets/d/1PT7GDSb56HKSjGbvmj5-bGVWCmbP8Wo7-tSkRxEk7aQ/edit). The CSV is a frozen snapshot, not an API-derived dataset. It has 15 wins, 10 losses and **zero draws**. All games are on Pinhole Punt. This snapshot is intentionally not represented as diverse or statistically powerful.

## Run

From this assignment folder (using the repo's existing dependencies):

```bash
pip install -r ../../requirements.txt
python -m unittest discover -s tests
jupyter notebook notebooks/brawl_stars_pipeline.ipynb
```

In Python, import `from src.features import ...` and `from src.models import ...` while working in this assignment folder.

## Experiments

- Class-frequency prior baseline
- Blue_1-only logistic regression (proxy, **not** the verified user character)
- Regularized signed-difference composition logistic regression
- Regularized side-separated logistic regression

Five-fold stratified **out-of-fold** predictions; log loss is the primary metric; report Brier, accuracy and balanced accuracy. Preprocessing and feature order are deterministic, with vocabulary based on brawler names only. No training on result/trophy metadata.

## Multinomial extension

Future labels are WIN, DRAW, LOSS. Do not infer draw probabilities from this dataset: no draws were observed. After genuine draw examples exist in reasonable numbers, fit a three-class multinomial model with softmax and evaluate multiclass log loss.

## Integrity notes

The team positions are left/right in screenshots, assumed to correspond to Blue/Red. Individual portrait transcription needs final manual verification. **Do not interpret a good score as causal draft strength**: small n, patch variation, matchmaking and repeated player habits are confounders.

Full execution plan: [Google Doc](https://docs.google.com/document/d/1IPTG9PpUdT5H7X-I0S7bjCjKbauvxZaRpzkvWXfPYQ0/edit).
