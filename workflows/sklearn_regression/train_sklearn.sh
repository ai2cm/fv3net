python -m fv3net.regression.sklearn \
  gs://vcm-ml-data/test-annak/2020-04-08_test_bigzarr_train_pipeline/ \
  workflows/end_to_end/kustomization/train_sklearn_model.yml \
  gs://vcm-ml-data/test-annak/2020-04-08_sklearn_train
