"""TF-IDF and Multinomial Naive Bayes utilities from CS156 Session 5."""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


SMS_URL = (
    "https://raw.githubusercontent.com/"
    "milindsoorya/Spam-Classifier-in-python/main/dataset/spam.csv"
)


def load_sms_data(url: str = SMS_URL):
    """Load the SMS spam dataset used in the PCW."""
    df = pd.read_csv(url, encoding="latin-1")
    return df, df["v2"], df["v1"]


def vectorize_text(texts):
    """Fit TF-IDF and return the sparse feature matrix and vectorizer."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def train_naive_bayes(X, labels):
    """Fit a Multinomial Naive Bayes classifier."""
    model = MultinomialNB()
    model.fit(X, labels)
    return model


def ordered_tokens(vectorizer):
    """Return vocabulary tokens in feature-column order."""
    return np.array(
        [
            token
            for token, index in sorted(
                vectorizer.vocabulary_.items(),
                key=lambda item: item[1],
            )
        ]
    )


def feature_probabilities(model, vectorizer, class_name: str):
    """Return tokens and P(feature | class) values."""
    class_idx = list(model.classes_).index(class_name)
    tokens = ordered_tokens(vectorizer)
    probabilities = np.exp(model.feature_log_prob_[class_idx])
    return tokens, probabilities


def class_log_odds(model, vectorizer, positive="spam", negative="ham"):
    """Return tokens and log P(feature|positive) - log P(feature|negative)."""
    positive_idx = list(model.classes_).index(positive)
    negative_idx = list(model.classes_).index(negative)

    tokens = ordered_tokens(vectorizer)
    log_odds = (
        model.feature_log_prob_[positive_idx]
        - model.feature_log_prob_[negative_idx]
    )
    return tokens, log_odds
