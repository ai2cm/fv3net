python -m fv3net.diagnostics.sklearn_model_performance \
  20200205.205016_model_training_files/20200205.205016_sklearn_model.pkl \
  gs://vcm-ml-data/test-annak/2020-02-20_train_data_pipeline_downsampled/test \
  gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/C48_gfsphysics_15min_coarse.zarr \
  gs://vcm-ml-public/test-annak/test-new-diags \
  48
