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
    config.resolve_output_yaml(tmp_path)
    assert os.path.isfile(tmp_path / config_file), f"{config_file} was not created"


config_files = ["scream_output_coarsening_2d.yaml", "scream_output_coarsening_3d.yaml"]


def test_resolve_multiple_output_yaml(tmp_path):
    config = scream_run.ScreamConfig()
    config.output_yaml = [
        os.path.join(output_bucket_dir, filename) for filename in config_files
    ]
    config.resolve_output_yaml(tmp_path)
    for filename in config_files:
        assert os.path.isfile(tmp_path / filename), f"{filename} was not created"
