"""Text cleaning for tweets.

The pipeline is deliberately dependency-light (no NLTK corpus downloads) so it
runs the same way locally, in CI and inside a serialized scikit-learn model.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations flip sentiment ("not good"), so they must survive stop-word removal.
NEGATIONS = frozenset({"no", "not", "nor", "never", "none", "nothing", "nobody", "cannot"})
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NEGATION_CONTRACTION_RE = re.compile(r"\b(\w+)n't\b")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")  # "soooo" -> "soo", keeps real doubles like "good"
_WHITESPACE_RE = re.compile(r"\s+")

_stemmer = PorterStemmer()


def clean_tweet(text: str, *, remove_stopwords: bool = True, stem: bool = True) -> str:
    """Normalize a raw tweet into a space-separated string of tokens."""
    text = html.unescape(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = _NEGATION_CONTRACTION_RE.sub(r"\1 not", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)

    tokens = [t for t in text.split() if len(t) > 1 or t in {"i", "a"}]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return _WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()


class TweetPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapping :func:`clean_tweet`.

    Putting cleaning inside the pipeline means a saved model accepts raw tweets.
    """

    def __init__(self, remove_stopwords: bool = True, stem: bool = True):
        self.remove_stopwords = remove_stopwords
        self.stem = stem

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [
            clean_tweet(str(x), remove_stopwords=self.remove_stopwords, stem=self.stem) for x in X
        ]
