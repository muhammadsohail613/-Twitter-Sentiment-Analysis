#!/usr/bin/env bash
# Download the Sentiment140 training set (1.6M tweets, ~230 MB unzipped) into data/.
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
URL="https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
CSV="training.1600000.processed.noemoticon.csv"

mkdir -p "$DATA_DIR"
if [[ -f "$DATA_DIR/$CSV" ]]; then
  echo "Dataset already present: $DATA_DIR/$CSV"
  exit 0
fi

echo "Downloading Sentiment140..."
curl -fL "$URL" -o "$DATA_DIR/sentiment140.zip"
unzip -o "$DATA_DIR/sentiment140.zip" "$CSV" -d "$DATA_DIR"
rm "$DATA_DIR/sentiment140.zip"
echo "Saved to $DATA_DIR/$CSV"
