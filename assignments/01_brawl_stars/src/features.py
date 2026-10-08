"""Deterministic composition features. No outcome information is used."""
import numpy as np
import pandas as pd

BLUE = ["Blue_1", "Blue_2", "Blue_3"]
RED = ["Red_1", "Red_2", "Red_3"]
SLOTS = BLUE + RED

def validate_matches(df):
    required = ["Match_ID", *SLOTS, "Blue_Win"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[required].isnull().any().any():
        raise ValueError("Missing core match fields")
    if df.Match_ID.duplicated().any():
        raise ValueError("Duplicate Match_ID")
    if not set(df.Blue_Win.astype(int)).issubset({0, 1}):
        raise ValueError("Invalid outcomes")
    for side in (BLUE, RED):
        if any(len(set(row)) != 3 for row in df[side].to_numpy()):
            raise ValueError("Duplicate character within team")
    if "Result" in df:
        truth = df.Result.str.upper().map({"VICTORY":1,"DEFEAT":0})
        if truth.isna().any() or not (truth.to_numpy() == df.Blue_Win.astype(int).to_numpy()).all():
            raise ValueError("Inconsistent outcome / result")
    return True

def normalize_names(df):
    out = df.copy()
    for c in SLOTS:
        out[c] = out[c].astype(str).str.strip().str.upper().str.replace(r"\\s+", "_", regex=True)
        out[c] = out[c].replace({"COLLETTE": "COLETTE"})
    return out

def vocabulary(df):
    return sorted(set(df[SLOTS].to_numpy().ravel()))

def encode(df, vocab, representation="signed"):
    """Produce columns in fixed vocab order; unknown brawlers trigger an error."""
    vocab = list(vocab)
    index = {c: i for i, c in enumerate(vocab)}
    n = len(df)
    x = np.zeros((n, len(vocab) if representation == "signed" else len(vocab)*2), dtype=float)
    for i, row in enumerate(df.itertuples(index=False)):
        d = row._asdict()
        for c in BLUE:
            k = index[d[c]]
            x[i, k] += 1
        for c in RED:
            k = index[d[c]]
            if representation == "signed":
                x[i, k] -= 1
            elif representation == "separate":
                x[i, len(vocab)+k] += 1
            else:
                raise ValueError("representation must be signed or separate")
    return x
