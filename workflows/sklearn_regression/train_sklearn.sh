python -m fv3net.regression.sklearn.train \
  --train-config-file example_rf_training_config.yml \
  --output-dir-suffix sklearn_regression \
  --train-data-path gs://vcm-ml-data/test_annak/2020-02-05_train_data_pipeline/train \
  --remote-output-url gs://vcm-ml-data/test_annak/
