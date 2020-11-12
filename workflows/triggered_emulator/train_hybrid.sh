#!/bin/bash
set -e
set -x

DATA=/Users/noah/data/dev/2020-11-11-hybrid-data-resampled
OUTPUT=gs://vcm-ml-archive/noah/emulator/$(date +%F)-triggered-hybrid
CLASSIFIER=hybrid_classifier.pkl
REGRESSOR=hybrid_regressor.pkl

[[ -d "$DATA" ]] || python download_hybrid.py "$DATA"
[[ -f "$CLASSIFIER" ]] || python train_classifier.py "$DATA" "$CLASSIFIER"
[[ -f "$REGRESSOR" ]] || python train_regressor.py "$DATA" "$REGRESSOR"

# upload model
gsutil cp "$REGRESSOR" "$OUTPUT/regressor.pkl"
gsutil cp "$CLASSIFIER" "$OUTPUT/classifier.pkl"
gsutil cp hybrid-mapper.yaml "$OUTPUT/hybrid-mapper.yaml"
