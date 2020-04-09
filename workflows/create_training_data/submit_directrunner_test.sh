python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/test-annak/2020-03-27_test_one_step/big.zarr \
gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
gs://vcm-ml-data/test-annak/2020-03-27_test_bigzarr_train_pipeline \
./pipelines/create_training_data/variable_names.yml \
  --runner DirectRunner
