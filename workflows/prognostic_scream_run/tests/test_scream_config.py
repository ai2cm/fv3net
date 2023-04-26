import os
import scream_run
import yaml
import pytest
from typing import List

dirname = os.path.dirname(os.path.abspath(__file__))

EXAMPLE_CONFIGS_DIR = os.path.join(dirname, "example_configs")


@pytest.mark.parametrize(
    "path, file", [(EXAMPLE_CONFIGS_DIR, "scream_ne30pg2.yaml")],
)
def test_example_config_can_initialize(path: str, file: str):
    with open(os.path.join(path, file), "r") as f:
        config = scream_run.ScreamConfig.from_dict(yaml.safe_load(f))
    assert isinstance(config, scream_run.ScreamConfig)


output_bucket_dir = "gs://vcm-scream/config/output"


@pytest.mark.parametrize(
    "path, filename", [pytest.param(output_bucket_dir, "default.yaml"),],
)
def test_resolve_single_output_yaml(path: str, filename: str):
    config = scream_run.ScreamConfig()
    config.output_yaml = os.path.join(path, filename)
    config.resolve_output_yaml("/tmp")
    assert os.path.isfile(f"/tmp/{filename}"), f"{filename} was not created"


@pytest.mark.parametrize(
    "path, file_list",
    [
        pytest.param(
            output_bucket_dir,
            ["scream_output_coarsening_2d.yaml", "scream_output_coarsening_3d.yaml"],
        ),
    ],
)
def test_resolve_multiple_output_yaml(path: str, file_list: List[str]):
    config = scream_run.ScreamConfig()
    config.output_yaml = [os.path.join(path, filename) for filename in file_list]
    config.resolve_output_yaml("/tmp")
    for filename in file_list:
        assert os.path.isfile(f"/tmp/{filename}"), f"{filename} was not created"
