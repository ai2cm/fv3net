python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/2020-02-28-X-SHiELD-2019-12-02-deep-and-mp-off \
gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
gs://vcm-ml-data/experiments-2020-03/deep-conv-mp-off/train-test-data \
  --runner DirectRunner
