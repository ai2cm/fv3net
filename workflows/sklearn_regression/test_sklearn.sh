python -m fv3net.diagnostics.sklearn_model_performance \
  gs://vcm-ml-data/orchestration-testing/test-experiment-7021f96d/train_sklearn_model_train-config-file_example_base_rf_training_config.yml \
  gs://vcm-ml-data/orchestration-testing/test-experiment-0ec4a4b1/create_training_data_train-fraction_0.5_runner_DataflowRunner \
  gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/C48_gfsphysics_15min_coarse.zarr \
  gs://vcm-ml-public/test-annak/test-new-diags \
  --num_test_zarrs 4
