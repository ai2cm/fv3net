#!/bin/bash

set -e

EXPERIMENT=2020-11-09-nudge-to-obs-NN-trials-test
mkdir -p configs

for i in {2..2}
do

    sed '/random_seed/d' configs/base-training-config.yml > configs/training-config-seed-$i-test.yml
    echo "random_seed: $i" >> configs/training-config-seed-$i-test.yml

    argo submit \
        --from workflowtemplate/train-diags-prog \
        -p root=gs://vcm-ml-scratch/annak/$EXPERIMENT/seed-$i \
        -p train-test-data=gs://vcm-ml-experiments/2020-10-30-nudge-to-obs-GRL-paper/nudge-to-obs-run-3hr-diags \
        -p training-config="$(< configs/training-config-seed-$i-test.yml)" \
        -p train-routine="keras" \
        -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
        -p initial-condition="20160101.000000" \
        -p prognostic-run-config="$(< configs/prognostic-run-test.yml)" \
        -p train-times="$(<  ../train_short.json)" \
        -p test-times="$(<  ../test_short.json)" \
        -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT/seed-$i \
        -p cpu-prog=24 \
        -p memory-prog="25Gi" \
        -p flags="--nudge-to-observations" \
        -p chunks="$(< chunks.yaml)" \
        --name nudge-to-obs-nn-trials-seed-$i-test
done