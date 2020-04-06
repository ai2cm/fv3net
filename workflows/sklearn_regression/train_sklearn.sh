python -m fv3net.regression.sklearn.train \
  gs://vcm-ml-data/test-annak/2020-04-03_test_bigzarr_train_pipeline \
  workflows/sklearn_regression/example_rf_training_config.yml \
  gs://vcm-ml-data/test-annak/2020-04-06_sklearn_train
