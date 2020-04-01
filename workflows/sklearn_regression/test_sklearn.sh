python -m fv3net.diagnostics.sklearn_model_performance \
  gs://vcm-ml-data/experiments-2020-03/deep-conv-off-mp-off-8c96d2ad/train_sklearn_model_train-config-file_example_base_rf_training_config.yml/ \
  gs://vcm-ml-data/experiments-2020-03/deep-conv-off-mp-off-8c96d2ad/create_training_data_timesteps-per-output-file_1_train-fraction_0.5/ \
  gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
  gs://vcm-ml-public/test-annak/test-new-diags \
  --num_test_zarrs 4
