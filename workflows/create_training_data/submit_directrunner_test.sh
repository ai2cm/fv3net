python -m fv3net.pipelines.create_training_data \
 --gcs-input-data-path 2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/one_step_output/C48 \
 --diag-c384-path gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/gfsphysics_15min_coarse.zarr \
 --gcs-output-data-dir test-annak/2020-01-29_train_data_pipeline	\
  --gcs-bucket gs://vcm-ml-data \
  --runner DirectRunner