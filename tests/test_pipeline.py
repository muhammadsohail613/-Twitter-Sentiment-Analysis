import csv
import json

import pytest

from sentiment.data import load_sentiment140
from sentiment.models import CLASSIFIERS, build_pipeline
from sentiment.predict import predict
from sentiment.train import main as train_main

POSITIVE = [
    "I love this, what a great day",
    "so happy with the amazing results",
    "best concert ever, loved it",
    "great food and awesome friends",
    "feeling wonderful and happy today",
]
NEGATIVE = [
    "I hate this, what a terrible day",
    "so sad about the awful results",
    "worst concert ever, hated it",
    "bad food and horrible service",
    "feeling miserable and sad today",
]


@pytest.fixture
def sentiment140_csv(tmp_path):
    """A tiny file in the exact Sentiment140 format (label-sorted, no header)."""
    path = tmp_path / "sentiment140.csv"
    with path.open("w", newline="", encoding="ISO-8859-1") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        i = 0
        for label, texts in ((0, NEGATIVE), (4, POSITIVE)):
            for _ in range(10):
                for text in texts:
                    writer.writerow(
                        [label, i, "Mon Apr 06 22:19:45 PDT 2009", "NO_QUERY", "u", text]
                    )
                    i += 1
    return path


def test_load_maps_labels_and_balances_sample(sentiment140_csv):
    df = load_sentiment140(sentiment140_csv, sample_size=40)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 40
    assert df["label"].value_counts().to_dict() == {0: 20, 1: 20}


@pytest.mark.parametrize("name", list(CLASSIFIERS))
def test_every_model_learns_toy_data(name):
    texts, labels = POSITIVE * 4 + NEGATIVE * 4, [1] * 20 + [0] * 20
    model = build_pipeline(name).fit(texts, labels)
    assert (model.predict(texts) == labels).mean() >= 0.9


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("gpt")


def test_train_then_predict_end_to_end(sentiment140_csv, tmp_path):
    reports, models = tmp_path / "reports", tmp_path / "models"
    train_main(
        [
            "--data",
            str(sentiment140_csv),
            "--sample-size",
            "0",
            "--models",
            "logistic_regression",
            "naive_bayes",
            "--output",
            str(reports),
            "--model-dir",
            str(models),
        ]
    )

    metrics = json.loads((reports / "metrics.json").read_text())
    assert {r["key"] for r in metrics["results"]} == {"logistic_regression", "naive_bayes"}
    assert (reports / "model_comparison.png").exists()

    labels = predict(
        ["what a great amazing day", "awful terrible service"], models / "sentiment_model.joblib"
    )
    assert labels == ["Positive", "Negative"]
