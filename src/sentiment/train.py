"""Train and compare sentiment classifiers.

Example:
    python -m sentiment.train --data data/training.1600000.processed.noemoticon.csv \
        --sample-size 200000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from sentiment.data import LABEL_NAMES, load_sentiment140  # noqa: E402
from sentiment.models import CLASSIFIERS, DISPLAY_NAMES, build_pipeline  # noqa: E402


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    pct = cm / cm.sum()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), count in np.ndenumerate(cm):
        color = "white" if count > cm.max() / 2 else "black"
        ax.text(j, i, f"{count:,}\n{pct[i, j]:.1%}", ha="center", va="center", color=color)
    ax.set_xticks([0, 1], LABEL_NAMES)
    ax.set_yticks([0, 1], LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results: pd.DataFrame, path: Path) -> None:
    ordered = results.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(ordered["model"], ordered["accuracy"], color="#2b6cb0")
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in ordered["accuracy"]], padding=4)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Test accuracy")
    ax.set_title("Model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path, help="Path to the Sentiment140 CSV")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Balanced subsample size (default: 200000; use 0 for all 1.6M)",
    )
    parser.add_argument("--models", nargs="+", default=list(CLASSIFIERS), choices=list(CLASSIFIERS))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    df = load_sentiment140(args.data, sample_size=args.sample_size or None, random_state=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )
    print(f"Train: {len(X_train):,} tweets | Test: {len(X_test):,} tweets")

    rows, best = [], None
    for name in args.models:
        pipeline = build_pipeline(name)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        y_pred = pipeline.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        rows.append(
            {
                "model": DISPLAY_NAMES[name],
                "key": name,
                **metrics,
                "train_seconds": round(train_seconds, 2),
            }
        )
        print(
            f"{DISPLAY_NAMES[name]:<24} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"({train_seconds:.1f}s)"
        )

        plot_confusion_matrix(
            y_test, y_pred, DISPLAY_NAMES[name], args.output / f"confusion_matrix_{name}.png"
        )
        if best is None or metrics["accuracy"] > best[1]:
            best = (name, metrics["accuracy"], pipeline)

    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    results.to_csv(args.output / "metrics.csv", index=False)
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "results": results.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    plot_comparison(results, args.output / "model_comparison.png")

    best_name, best_acc, best_pipeline = best
    joblib.dump(best_pipeline, args.model_dir / "sentiment_model.joblib")
    print(
        f"\nBest model: {DISPLAY_NAMES[best_name]} ({best_acc:.2%}) "
        f"-> {args.model_dir / 'sentiment_model.joblib'}"
    )
    return results


if __name__ == "__main__":
    main()
