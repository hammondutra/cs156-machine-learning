"""CS156 Session 21 PCW: contrastive projection + Ames housing extension.

Run: python src/session21_assignment2_prep.py [--from-raw] [--no-show]

The contrastive exercise is a pedagogical NumPy projection, NOT full SetFit:
no pretrained sentence transformer or classifier is trained. Labels are synthetic
and reported metrics are training-pair metrics, not held-out security scores.
The housing section uses a committed summary offline; --from-raw repeats the
cleaning and aggregation on the 1,460-sale public Ames training CSV.
"""
# %% [contrastive_start]
import re
import numpy as np

ALLOWED = [
    "User requests a repo summary. Agent reads README.md.",
    "User requests a repo summary. Agent reads docs/guide.md.",
    "User requests tests. Agent runs pytest.",
    "User requests a code fix. Agent edits src/helper.py.",
    "User requests a report. Agent reads reports/monthly.csv.",
    "User approves staging deployment. Agent deploys to staging.",
    "User requests log review. Agent reads logs/app.log.",
    "User requests a documentation fix. Agent edits docs/setup.md.",
]
UNSAFE = [
    "User requests a repo summary. Agent reads .env secrets.",
    "User requests a repo summary. Agent uploads keys to external host.",
    "User requests tests. Agent deletes the database.",
    "User requests a code fix. Agent edits production permissions.",
    "User requests a report. Agent exports private customer records.",
    "User approves staging deployment. Agent deploys to production.",
    "User requests log review. Agent turns off audit logging.",
    "User requests a documentation fix. Agent modifies admin accounts.",
]

def contrastive_objective(X, W, pairs):
    """Pairwise cosine-MSE teaching loss with analytic gradient."""
    loss = 0.0
    gradient = np.zeros_like(W)
    similarities = {0: [], 1: []}
    for i, j, target in pairs:
        a, b = X[i] @ W, X[j] @ W
        na, nb = np.linalg.norm(a) + 1e-8, np.linalg.norm(b) + 1e-8
        za, zb = a / na, b / nb
        sim = float(za @ zb)
        similarities[int(target)].append(sim)
        error = sim - target
        loss += error ** 2
        grad_a = (zb - sim * za) / na
        grad_b = (za - sim * zb) / nb
        gradient += 2 * error * (
            np.outer(X[i], grad_a) + np.outer(X[j], grad_b)
        )
    n = len(pairs)
    return loss / n, gradient / n, {
        k: float(np.mean(v)) for k, v in similarities.items()
    }

def run_contrastive(steps=300, learning_rate=0.5):
    texts = ALLOWED + UNSAFE
    labels = np.array([0] * len(ALLOWED) + [1] * len(UNSAFE))
    tokens = [set(re.findall(r"[a-z0-9_./-]+", t.lower())) for t in texts]
    vocab = sorted(set().union(*tokens))
    X = np.array([[float(word in row) for word in vocab] for row in tokens])
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    pairs = [(i, j, float(labels[i] == labels[j]))
             for i in range(len(texts)) for j in range(i + 1, len(texts))]
    W = np.random.default_rng(7).normal(0, 0.2, (len(vocab), 12))
    before_loss, _, before = contrastive_objective(X, W, pairs)
    losses = []
    for _ in range(steps):
        loss, gradient, _ = contrastive_objective(X, W, pairs)
        losses.append(loss)
        W -= learning_rate * gradient
    after_loss, _, after = contrastive_objective(X, W, pairs)
    print(f"Synthetic examples: {len(texts)} | unique pairs: {len(pairs)}")
    print(f"Training loss: {before_loss:.3f} -> {after_loss:.3f}")
    print(f"Same-label similarity: {before[1]:.3f} -> {after[1]:.3f}")
    print(f"Different-label similarity: {before[0]:.3f} -> {after[0]:.3f}")
    print("Training pairs only: NOT an unseen security-model evaluation.")
    return {"before": before, "after": after, "losses": losses,
            "initial_loss": before_loss, "final_loss": after_loss}

# %% [contrastive_end]
# %% [ames_start]
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

AMES_URL = "https://raw.githubusercontent.com/knaat/Ames-housing/main/train.csv"
SUMMARY_NAME = "session21_ames_neighborhood_summary.csv"

def find_summary_csv():
    candidates = [Path("data") / SUMMARY_NAME,
                  Path("../data") / SUMMARY_NAME]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent.parent /
                          "data" / SUMMARY_NAME)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {SUMMARY_NAME}. Run from repo root.")

def summarize_ames(from_raw=False):
    if from_raw:
        data = pd.read_csv(AMES_URL)
        data["Neighborhood"] = data["Neighborhood"].astype("string").str.strip()
        data["SalePrice"] = pd.to_numeric(data["SalePrice"], errors="coerce")
        data = data.dropna(subset=["Neighborhood", "SalePrice"])
        data = data[data["SalePrice"] > 0]
        summary = data.groupby("Neighborhood")["SalePrice"].agg(
            n="size", median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        ).reset_index()
        origin = f"Raw Kaggle mirror, {len(data)} valid sales"
    else:
        summary = pd.read_csv(find_summary_csv())
        origin = "Verified, aggregated CSV in data/ (offline)"
    summary = summary.sort_values("median", ascending=False).reset_index(drop=True)
    print(f"Ames source: {origin}")
    print(f"{len(summary)} neighborhoods; {summary['n'].sum()} sales")
    print(summary[["Neighborhood", "n", "median"]].head(3).to_string(index=False))
    return summary

def plot_ames(summary, show=True, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(summary))
    ax.hlines(y, summary["q25"] / 1000, summary["q75"] / 1000,
              linewidth=5, alpha=0.5, label="Middle 50% of sales")
    ax.scatter(summary["median"] / 1000, y, zorder=3, label="Median")
    ax.set_yticks(y, summary["Neighborhood"])
    ax.invert_yaxis()
    ax.set_xlabel("Sale price (thousands of USD)")
    ax.set_title("Ames, Iowa — historical home sale prices")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
# %% [ames_end]

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-plot", default=None)
    args = parser.parse_args()
    run_contrastive()
    print()
    summary = summarize_ames(from_raw=args.from_raw)
    plot_ames(summary, show=not args.no_show, save_path=args.save_plot)
    top, bottom = summary.iloc[0], summary.iloc[-1]
    print(f"Highest median: {top['Neighborhood']} USD {top['median']:,.0f}"
          f" (n={int(top['n'])})")
    print(f"Lowest median: {bottom['Neighborhood']} USD {bottom['median']:,.0f}"
          f" (n={int(bottom['n'])})")
    print("Differences are descriptive; neighborhood is not a causal explanation.")

if __name__ == "__main__":
    main()
