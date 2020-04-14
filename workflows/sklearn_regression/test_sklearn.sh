python -m fv3net.diagnostics.sklearn_model_performance \
  gs://vcm-ml-data/test-annak/2020-04-08_sklearn_train \
  gs://vcm-ml-data/test-annak/2020-04-08_test_bigzarr_train_pipeline  \
  gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
  gs://vcm-ml-public/test-annak/test-new-diags \
  ./fv3net/diagnostics/sklearn_model_performance/variable_names.yml \
  --num_test_zarrs 4
