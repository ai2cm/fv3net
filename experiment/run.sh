#!/bin/bash
set -e


IMAGE_TAG=42e3dbcc5995c4dce9c71c93ff60c489ba0245b7
EXPERIMENT_PREFIX=2020-10-09-nn-random-seed-test
mkdir -p training_configs

kubectl apply -k fv3net/workflows/argo

for i in {0..5}
do
  sed '/random_seed/d' train_config.yml > training_configs/test_altered_config_seed_$i.yml
  echo "random_seed: $i" >> training_configs/test_altered_config_seed_$i.yml
  argo submit \
    --from workflowtemplate/train-diags-prog \
    -p image-tag=IMAGE_TAG \
    -p root=gs://vcm-ml-scratch/annak/$EXPERIMENT/seed-$i \
    -p initial-condition="20160805.000000" \
    -p train-test-data='gs://vcm-ml-experiments/2020-09-24-tke-edmf-nudge-to-fine2' \
    -p training-config="$(< training_configs/test_altered_config_seed_$i.yml)" \
    -p train-routine="keras" \
    -p train-times="$(< train_short.json)" \
    -p test-times="$(< test_short.json)" \
    -p reference-restarts=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
    -p prognostic-run-config="$(< prognostic_config.yml)" \
    -p public-report-output=gs://vcm-ml-public/offline_ml_diags/$EXPERIMENT_PREFIX/seed-$i \
    -p segment-count=1 \
    --name nudge-to-fine-tke-edmf-nn-seed-$i
done
