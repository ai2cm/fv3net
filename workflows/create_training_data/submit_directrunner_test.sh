python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/2020-02-28-X-SHiELD-2019-12-02-deep-and-mp-off \
gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/C48_gfsphysics_15min_coarse.zarr \
gs://vcm-ml-data/test-annak/2020-03-07_test-onestep-diag-addition	\
  --runner DirectRunner
