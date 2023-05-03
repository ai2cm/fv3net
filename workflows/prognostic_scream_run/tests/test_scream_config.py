import os
import scream_run
import yaml

dirname = os.path.dirname(os.path.abspath(__file__))

EXAMPLE_CONFIGS_DIR = os.path.join(dirname, "example_configs")


def test_example_config_can_initialize():
    with open(os.path.join(EXAMPLE_CONFIGS_DIR, "scream_ne30pg2.yaml"), "r") as f:
        config = scream_run.ScreamConfig.from_dict(yaml.safe_load(f))
    assert isinstance(config, scream_run.ScreamConfig)


output_bucket_dir = "gs://vcm-scream/config/output"


config_file = "default.yaml"


def test_resolve_single_output_yaml(tmp_path):
    config = scream_run.ScreamConfig()
    config.output_yaml = os.path.join(output_bucket_dir, config_file)
    local_output_yaml = config.get_local_output_yaml(tmp_path)
    for filename in local_output_yaml:
        assert os.path.isfile(filename), f"{config_file} was not created"


config_files = ["scream_output_coarsening_2d.yaml", "scream_output_coarsening_3d.yaml"]


def test_resolve_multiple_output_yaml(tmp_path):
    config = scream_run.ScreamConfig()
    config.output_yaml = [
        os.path.join(output_bucket_dir, filename) for filename in config_files
    ]
    local_output_yaml = config.get_local_output_yaml(tmp_path)
    for filename in local_output_yaml:
        assert os.path.isfile(filename), f"{filename} was not created"


def test_compose_write_scream_run_directory_command(tmp_path):
    config = scream_run.ScreamConfig()
    command = config.compose_write_scream_run_directory_command(tmp_path)
    expected_command = f"{tmp_path}/run_eamxx.sh --output_yaml {tmp_path}/default.yaml \
        --initial_conditions_type local \
        --create_newcase True --case_setup True --case_build True \
        --number_of_processers 16 --CASE_ROOT  --CASE_NAME scream_test \
        --COMPSET F2010-SCREAMv1 --RESOLUTION ne30pg2_ne30pg2 --ATM_NCPL 48 \
        --STOP_OPTION nhours --STOP_N 1 --REST_OPTION nhours --REST_N 1 \
        --HIST_OPTION ndays --HIST_N 1 --RUN_STARTDATE 2010-01-01 \
        --MODEL_START_TYPE initial --OLD_EXECUTABLE "
    assert " ".join(command.split()) == " ".join(expected_command.split())
