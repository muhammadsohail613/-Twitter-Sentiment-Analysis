"""Model definitions: each model is a full raw-text -> label pipeline."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment.preprocessing import TweetPreprocessor

CLASSIFIERS: dict[str, Callable[[], object]] = {
    "logistic_regression": lambda: LogisticRegression(C=2, max_iter=1000),
    "linear_svm": lambda: LinearSVC(C=0.5),
    "naive_bayes": lambda: BernoulliNB(),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "naive_bayes": "Bernoulli Naive Bayes",
    "knn": "K-Nearest Neighbors",
}


def build_pipeline(name: str, max_features: int = 500_000) -> Pipeline:
    """Return an unfitted preprocessing + TF-IDF + classifier pipeline."""
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(CLASSIFIERS)}")
    return Pipeline(
        [
            ("clean", TweetPreprocessor()),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=2)),
            ("clf", CLASSIFIERS[name]()),
        ]
    )
