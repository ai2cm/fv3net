#!/bin/bash
set -e
set -x

DATA=/Users/noah/data/dev/2020-11-11-hybrid-data-resampled
CLASSIFIER=hybrid_classifier.pkl
REGRESSOR=hybrid_regressor.pkl

python download_hybrid.py "$DATA"
python train_classifier.py "$DATA" "$CLASSIFIER"
python train_regressor.py "$DATA" "$REGRESSOR"