python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/test-annak/2020-04-03_onestep_zarr/big.zarr \
gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
gs://vcm-ml-data/test-annak/2020-04-07_test_bigzarr_train_pipeline \
./fv3net/pipelines/create_training_data/variable_names.yml \
  --runner DirectRunner
