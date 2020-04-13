python -m fv3net.pipelines.create_training_data \
gs://vcm-ml-data/test-brianh/one-step-run-883930c6/one_step_run_experiment_yaml_all-physics-off.yml_docker_image_prognostic_run:v0.1.0-a1_config-version_v0.3/big.zarr \
gs://vcm-ml-data/orchestration-testing/shield-coarsened-diags-2019-12-04 \
gs://vcm-ml-data/test-annak/2020-04-08_test_bigzarr_train_pipeline \
./fv3net/pipelines/create_training_data/variable_names.yml \
  --runner DirectRunner
