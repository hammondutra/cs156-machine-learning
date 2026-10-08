"""Out-of-fold model comparisons, shared folds, and honest probability metrics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, brier_score_loss
from .features import encode, vocabulary

def evaluate(df, random_state=42, n_splits=5):
    y = df.Blue_Win.astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("Binary comparison requires both outcomes")
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outputs = {}
    for kind in ("majority", "single_brawler", "signed_composition", "separate_sides"):
        probs = np.zeros(len(df), dtype=float)
        for tr, te in folds.split(df, y):
            train, test = df.iloc[tr], df.iloc[te]
            if kind == "majority":
                model = DummyClassifier(strategy="prior").fit(np.zeros((len(tr),1)), y[tr])
                p = model.predict_proba(np.zeros((len(te),1)))[:, list(model.classes_).index(1)]
            else:
                vocab = vocabulary(train)
                if kind == "single_brawler":
                    # A proxy feature: Blue_1, not verified to be the user's actual character.
                    xtr = np.array([[int(a==b) for b in vocab] for a in train.Blue_1])
                    xte = np.array([[int(a==b) for b in vocab] for a in test.Blue_1])
                else:
                    # Fixed roster for the current snapshot; no target-dependent feature selection.
                    all_vocab = vocabulary(df)
                    kind_rep = "signed" if kind == "signed_composition" else "separate"
                    xtr = encode(train, all_vocab, kind_rep)
                    xte = encode(test, all_vocab, kind_rep)
                model = LogisticRegression(C=0.1, max_iter=2000)
                model.fit(xtr, y[tr])
                p = model.predict_proba(xte)[:, list(model.classes_).index(1)]
            probs[te] = p
        predicted = (probs >= 0.5).astype(int)
        outputs[kind] = {
            "accuracy": float(accuracy_score(y,predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y,predicted)),
            "log_loss": float(log_loss(y,np.column_stack((1-probs, probs)),labels=[0,1])),
            "brier": float(brier_score_loss(y,probs)),
            "probabilities": probs.tolist(),
            "predicted": predicted.tolist()
        }
    return outputs
