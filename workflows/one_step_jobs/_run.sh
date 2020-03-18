fv3config=gs://vcm-ml-data/orchestration-testing/test-andrep/one_step_run_experiment_yaml_all-physics-off.yml_docker_image_fv3gfs-python:v0.3.1_config-version_v0.3/one_step_config/20160801.001500/fv3config.yml
fv3run --dockerimage us.gcr.io/vcm-ml/prognostic_run:0.1.0 $fv3config output_dir --runfile runfile.py
