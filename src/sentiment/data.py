"""Loading the Sentiment140 dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
# Sentiment140 encodes polarity as 0 = negative, 4 = positive.
LABEL_MAP = {0: 0, 4: 1}
LABEL_NAMES = ["Negative", "Positive"]


def load_sentiment140(
    path: str | Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the CSV and return a ``text`` / ``label`` frame.

    The raw file is sorted by label (800k negative, then 800k positive), so any
    subsample is stratified rather than taken from the top of the file.
    """
    df = pd.read_csv(path, encoding="ISO-8859-1", header=None, names=COLUMNS)
    df = df[df["target"].isin(LABEL_MAP)]
    df = pd.DataFrame({"text": df["text"].astype(str), "label": df["target"].map(LABEL_MAP)})

    if sample_size is not None and sample_size < len(df):
        per_class = sample_size // 2
        df = df.groupby("label", group_keys=False).sample(n=per_class, random_state=random_state)

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
