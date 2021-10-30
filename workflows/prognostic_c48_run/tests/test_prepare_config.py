import dataclasses

import dacite
import pytest
from runtime.segmented_run.prepare_config import (
    HighLevelConfig,
    UserConfig,
    instantiate_dataclass_from,
)
from runtime.diagnostics.fortran import FortranFileConfig
from runtime.segmented_run import prepare_config

TEST_DATA_DIR = "tests/prepare_config_test_data"


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            [f"{TEST_DATA_DIR}/prognostic_config.yml", "--model_url", "gs://ml-model"],
            id="ml",
        ),
        pytest.param([f"{TEST_DATA_DIR}/nudge_to_fine_config.yml"], id="n2f"),
        pytest.param([f"{TEST_DATA_DIR}/nudge_to_obs_config.yml"], id="n2o"),
        pytest.param([f"{TEST_DATA_DIR}/emulator.yml"], id="emulator"),
        pytest.param([f"{TEST_DATA_DIR}/fine_res_ml.yml"], id="fine-res-ml"),
    ],
)
def test_prepare_ml_config_regression(regtest, argv):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(argv)
    with regtest:
        prepare_config.prepare_config(args)


def test_get_user_config_is_valid():

    dict_ = {
        "base_version": "v0.5",
        "diagnostics": [
            {
                "name": "state_after_timestep.zarr",
                "times": {"frequency": 5400, "kind": "interval", "times": None},
                "variables": ["x_wind", "y_wind"],
            }
        ],
    }

    config = prepare_config.to_fv3config(dict_, model_url=[], diagnostic_ml=True,)
    # validate using dacite.from_dict
    dacite.from_dict(UserConfig, config)


def test_high_level_config_fortran_diagnostics():
    """Ensure that fortran diagnostics are translated to the Fv3config diag table"""
    config = HighLevelConfig(
        fortran_diagnostics=[FortranFileConfig(name="a", chunks={})]
    )
    dict_ = config.to_fv3config()
    # the chunk reading requires this to exist
    assert dict_["fortran_diagnostics"][0] == dataclasses.asdict(
        config.fortran_diagnostics[0]
    )
    assert len(dict_["diag_table"].file_configs) == 1


def test_instantiate_dataclass_from():
    @dataclasses.dataclass
    class A:
        a: int = 0

    @dataclasses.dataclass
    class B(A):
        b: int = 1

    b = B()
    a = instantiate_dataclass_from(A, b)
    assert a.a == b.a
    assert isinstance(a, A)
