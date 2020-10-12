#!/bin/bash
set -e


IMAGE_TAG=42e3dbcc5995c4dce9c71c93ff60c489ba0245b7
EXPERIMENT=2020-10-09-nn-random-seed
mkdir -p configs

for i in {0..2}
do
  sed '/random_seed/d' configs/train_config.yml > configs/config_seed_$i.yml
  echo "random_seed: $i" >> configs/config_seed_$i.yml
  argo submit \
    --from workflowtemplate/train-diags-prog \
    -p image-tag=$IMAGE_TAG \
    -p root=gs://vcm-ml-experiments/$EXPERIMENT/seed-$i \
    -p initial-condition="20160805.000000" \
    -p train-test-data='gs://vcm-ml-experiments/2020-09-24-tke-edmf-nudge-to-fine2' \
    -p training-config="$(< configs/config_seed_$i.yml)" \
    -p train-routine="keras" \
    -p train-times="$(< train.json)" \
    -p test-times="$(< test.json)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p prognostic-run-config="$(< configs/prognostic_config_full.yml)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT/seed-$i \
    -p segment-count=9 \
    --name test-nn-seed-$i
done
