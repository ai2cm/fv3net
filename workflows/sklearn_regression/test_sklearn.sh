python -m fv3net.regression.model_diagnostics \
  --test-data-path gs://vcm-ml-data/test-annak/2020-02-05_train_data_pipeline/test \
  --model-path 20200205.205016_model_training_files/20200205.205016_sklearn_model.pkl \
  --high-res-data-path gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/C48_gfsphysics_15min_coarse.zarr \
  --num-test-zarrs 4
