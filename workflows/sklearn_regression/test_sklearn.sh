python -m fv3net.diagnostics.sklearn_model_performance \
  gs://vcm-ml-data/test-annak/2020-04-08_sklearn_train \
  gs://vcm-ml-data/test-annak/2020-04-16_debug_intgr/create_training_data_  \
  gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
  ./fv3net/tests/end_to_end_integration/config/test_sklearn_variable_names.yml \
  gs://vcm-ml-public/test-annak/2020-04-17_scalarmetrics \
  --num_test_zarrs 2
