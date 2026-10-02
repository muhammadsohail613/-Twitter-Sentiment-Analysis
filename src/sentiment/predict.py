"""Classify tweets with a trained model.

Example:
    python -m sentiment.predict "I love this new phone!" "Worst service ever"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from sentiment.data import LABEL_NAMES

DEFAULT_MODEL = Path("models/sentiment_model.joblib")


def predict(texts: list[str], model_path: Path = DEFAULT_MODEL) -> list[str]:
    model = joblib.load(model_path)
    return [LABEL_NAMES[int(label)] for label in model.predict(texts)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("texts", nargs="+", help="One or more tweets to classify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    for text, label in zip(args.texts, predict(args.texts, args.model), strict=True):
        print(f"{label:<9} | {text}")


if __name__ == "__main__":
    main()
