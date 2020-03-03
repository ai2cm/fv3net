python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/one_step_output/C48 \
gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/C48_gfsphysics_15min_coarse.zarr \
gs://vcm-ml-data/test-annak/2020-01-29_train_data_pipeline	\
  --runner DirectRunner
