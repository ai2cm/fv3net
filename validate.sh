#!/bin/bash
set -e
set -x

[[ -f ./kustomize ]] || \
    ./install_kustomize.sh 3.8.6

kustomizations=("examples/train-evaluate-prognostic-run/" "examples/nudge-to-fine-run/")

for k in $kustomizations; do
    ./kustomize build $k
done
