import datetime
import json
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path

import cftime
import fv3config
import fv3fit
import numpy as np
import pytest
import runtime.metrics
import tensorflow as tf
import vcm.testing
import xarray as xr
import yaml
from machine_learning_mocks import get_mock_predictor
from runtime.names import FV3GFS_WRAPPER, SHIELD_WRAPPER
from testing_utils import requires_fv3gfs_wrapper, requires_shield_wrapper


FORTRAN_CONFIG_PATH = Path(__file__).parent / "regression_test_fortran_configs"
BASE_CONFIGS = {
    FV3GFS_WRAPPER: FORTRAN_CONFIG_PATH / "fv3gfs.yml",
    SHIELD_WRAPPER: FORTRAN_CONFIG_PATH / "shield.yml",
}
LOG_PATH = "logs.txt"
STATISTICS_PATH = "statistics.txt"
PROFILES_PATH = "profiles.txt"
CHUNKS_PATH = "chunks.yaml"


class ConfigEnum:
    nudging = "nudging"
    predictor = "predictor"
    microphys_emulation = "microphys_emulation"


# Necessary to know the number of restart timestamp folders to generate in fixture
START_TIME = [2016, 8, 1, 0, 0, 0]
TIMESTEP_MINUTES = 15
NUM_NUDGING_TIMESTEPS = 2
RUNTIME_MINUTES = TIMESTEP_MINUTES * NUM_NUDGING_TIMESTEPS
TIME_FMT = "%Y%m%d.%H%M%S"
RUNTIME = {"days": 0, "months": 0, "hours": 0, "minutes": RUNTIME_MINUTES, "seconds": 0}


def get_base_config(wrapper):
    fortran_config_file = BASE_CONFIGS[wrapper]
    with open(fortran_config_file, "r") as file:
        config = yaml.safe_load(file)

    # The default wrapper is fv3gfs.wrapper, so we only need to specify it if
    # using shield.wrapper.
    if wrapper == SHIELD_WRAPPER:
        config["wrapper"] = SHIELD_WRAPPER

    return config


def run_native(config, rundir):
    with tempfile.NamedTemporaryFile("w") as f:
        yaml.safe_dump(config, f)
        fv3_script = "runfv3"
        subprocess.check_call([fv3_script, "create", rundir, f.name])
        subprocess.check_call([fv3_script, "append", rundir])


def assets_from_initial_condition_dir(dir_: str):
    start = datetime.datetime(*START_TIME)  # type: ignore
    delta_t = datetime.timedelta(minutes=TIMESTEP_MINUTES)
    assets = []
    for i in range(NUM_NUDGING_TIMESTEPS + 1):
        timestamp = (start + i * delta_t).strftime(TIME_FMT)

        for tile in range(1, 7):
            for category in [
                "fv_core.res",
                "fv_srf_wnd.res",
                "fv_tracer.res",
                "phy_data",
                "sfc_data",
            ]:
                assets.append(
                    fv3config.get_asset_dict(
                        dir_,
                        f"{category}.tile{tile}.nc",
                        target_location=timestamp,
                        target_name=f"{timestamp}.{category}.tile{tile}.nc",
                    )
                )
    return assets


def _get_nudging_config(wrapper: str):
    config = get_base_config(wrapper)
    coupler_nml = config["namelist"]["coupler_nml"]
    coupler_nml["current_date"] = START_TIME
    coupler_nml.update(RUNTIME)

    config["nudging"] = {
        "restarts_path": ".",
        "timescale_hours": {"air_temperature": 3.0, "specific_humidity": 3.0},
    }

    timestamp_dir = config["initial_conditions"]
    config.setdefault("patch_files", []).extend(
        assets_from_initial_condition_dir(timestamp_dir)
    )
    if coupler_nml["dt_atmos"] // 60 != TIMESTEP_MINUTES:
        raise ValueError(
            "Model timestep in default_fv3config not aligned"
            " with specified module's TIMESTEP_MINUTES variable."
        )

    return config


def get_nudging_config(wrapper: str, tendencies_path: str):
    config = _get_nudging_config(wrapper)
    config["tendency_prescriber"] = {
        "mapper_config": {
            "function": "open_zarr",
            "kwargs": {"data_path": tendencies_path},
        },
        "variables": {"air_temperature": "Q1"},
    }
    config["diagnostics"] = [
        {
            "name": "diags.zarr",
            "times": {"kind": "interval", "frequency": 900, "times": None},
            "variables": [
                "air_temperature_reference",
                "air_temperature_tendency_due_to_nudging",
                "area",
                "cnvprcp_after_physics",
                "cnvprcp_after_python",
                "evaporation",
                "column_heating_due_to_nudging",
                "net_moistening_due_to_nudging",
                "physics_precip",
                "specific_humidity_reference",
                "specific_humidity_tendency_due_to_nudging",
                "storage_of_mass_due_to_fv3_physics",
                "storage_of_mass_due_to_python",
                "storage_of_specific_humidity_path_due_to_fv3_physics",
                "storage_of_specific_humidity_path_due_to_microphysics",
                "storage_of_specific_humidity_path_due_to_python",
                "storage_of_total_water_path_due_to_fv3_physics",
                "storage_of_total_water_path_due_to_python",
                "storage_of_internal_energy_path_due_to_python",
                "surface_temperature_reference",
                "tendency_of_air_temperature_due_to_fv3_physics",
                "tendency_of_air_temperature_due_to_python",
                "tendency_of_air_temperature_due_to_tendency_prescriber",
                "tendency_of_eastward_wind_due_to_fv3_physics",
                "tendency_of_eastward_wind_due_to_python",
                "tendency_of_northward_wind_due_to_fv3_physics",
                "tendency_of_northward_wind_due_to_python",
                "tendency_of_specific_humidity_due_to_fv3_physics",
                "tendency_of_specific_humidity_due_to_python",
                "tendency_of_internal_energy_due_to_python",
                "total_precip_after_physics",
                "total_precipitation_rate",
                "water_vapor_path",
            ],
        }
    ]
    config["fortran_diagnostics"] = []
    return config


def get_ml_config(wrapper, model_path):
    config = get_base_config(wrapper)
    config["diagnostics"] = [
        {
            "name": "diags.zarr",
            "times": {"kind": "interval", "frequency": 900, "times": None},
            "variables": [
                "air_temperature",
                "area",
                "cnvprcp_after_physics",
                "cnvprcp_after_python",
                "column_integrated_dQ1_change_non_neg_sphum_constraint",
                "column_integrated_dQ2_change_non_neg_sphum_constraint",
                "column_integrated_dQu_stress",
                "column_integrated_dQv_stress",
                "dQ1",
                "dQ2",
                "dQu",
                "dQv",
                "evaporation",
                "column_heating_due_to_machine_learning",
                "net_moistening_due_to_machine_learning",
                "physics_precip",
                "pressure_thickness_of_atmospheric_layer",
                "specific_humidity",
                "specific_humidity_limiter_active",
                "storage_of_mass_due_to_fv3_physics",
                "storage_of_mass_due_to_python",
                "storage_of_specific_humidity_path_due_to_fv3_physics",
                "storage_of_specific_humidity_path_due_to_microphysics",
                "storage_of_specific_humidity_path_due_to_python",
                "storage_of_total_water_path_due_to_fv3_physics",
                "storage_of_total_water_path_due_to_python",
                "storage_of_internal_energy_path_due_to_fv3_physics",
                "tendency_of_air_temperature_due_to_fv3_physics",
                "tendency_of_air_temperature_due_to_python",
                "tendency_of_eastward_wind_due_to_fv3_physics",
                "tendency_of_eastward_wind_due_to_python",
                "tendency_of_northward_wind_due_to_fv3_physics",
                "tendency_of_northward_wind_due_to_python",
                "tendency_of_specific_humidity_due_to_fv3_physics",
                "tendency_of_specific_humidity_due_to_python",
                "tendency_of_internal_energy_due_to_fv3_physics",
                "tendency_of_x_wind_due_to_python",
                "tendency_of_y_wind_due_to_python",
                "total_precip_after_physics",
                "total_precipitation_rate",
                "water_vapor_path",
            ],
        }
    ]
    config["fortran_diagnostics"] = []
    config["scikit_learn"] = {
        "model": [model_path],
        "use_mse_conserving_humidity_limiter": False,
    }
    return config


def get_emulation_config(wrapper: str, model_path: str):
    config = get_base_config(wrapper)

    config["zhao_carr_emulation"] = {
        "storage": {"save_nc": True, "save_zarr": True, "output_freq_sec": 900},
        "model": {"path": model_path},
    }

    physics = config["namelist"]["gfs_physics_nml"]
    physics["imp_physics"] = 99
    physics["ncld"] = 1
    physics["emulate_zc_microphysics"] = True
    physics["save_zc_microphysics"] = True
    physics["satmedmf"] = False
    physics["hybedmf"] = True

    fv_core = config["namelist"]["fv_core_nml"]
    fv_core["nwat"] = 2
    fv_core["do_sat_adj"] = False

    config["diagnostics"] = [
        {
            "name": "diags.zarr",
            "times": {"kind": "interval", "frequency": 900, "times": None},
            "variables": [
                "area",
                "cnvprcp_after_physics",
                "evaporation",
                "physics_precip",
                "storage_of_mass_due_to_fv3_physics",
                "storage_of_specific_humidity_path_due_to_fv3_physics",
                "storage_of_specific_humidity_path_due_to_microphysics",
                "storage_of_total_water_path_due_to_fv3_physics",
                "storage_of_internal_energy_path_due_to_fv3_physics",
                "tendency_of_air_temperature_due_to_fv3_physics",
                "tendency_of_eastward_wind_due_to_fv3_physics",
                "tendency_of_northward_wind_due_to_fv3_physics",
                "tendency_of_specific_humidity_due_to_fv3_physics",
                "tendency_of_internal_energy_due_to_fv3_physics",
                "total_precip_after_physics",
                "total_precipitation_rate",
                "water_vapor_path",
            ],
        }
    ]

    return config


def _tendency_dataset():
    temperature_tendency = np.full((6, 8, 63, 12, 12), 0.1 / 86400)
    times = [
        cftime.DatetimeJulian(2016, 8, 1) + timedelta(minutes=n)
        for n in range(0, 120, 15)
    ]
    da = xr.DataArray(
        data=temperature_tendency,
        dims=["tile", "time", "z", "y", "x"],
        coords=dict(time=times),
        attrs={"units": "K/s"},
    )
    return xr.Dataset({"Q1": da})


@pytest.fixture(
    scope="module",
    params=[
        ConfigEnum.predictor,
        ConfigEnum.nudging,
        pytest.param(ConfigEnum.microphys_emulation, marks=requires_fv3gfs_wrapper),
    ],
)
def configuration(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(FV3GFS_WRAPPER, marks=requires_fv3gfs_wrapper),
        pytest.param(SHIELD_WRAPPER, marks=requires_shield_wrapper),
    ],
)
def wrapper(request):
    return request.param


def create_emulation_model():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    out_ = tf.keras.layers.Lambda(lambda x: x + 1, name="air_temperature_dummy")(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)
    return model


@pytest.fixture(scope="module")
def completed_rundir(wrapper, configuration, tmpdir_factory):

    tendency_dataset_path = tmpdir_factory.mktemp("tendencies")

    if configuration == ConfigEnum.predictor:
        model = get_mock_predictor()
        model_path = str(tmpdir_factory.mktemp("model"))
        fv3fit.dump(model, str(model_path))
        config = get_ml_config(wrapper, model_path)
    elif configuration == ConfigEnum.nudging:
        tendencies = _tendency_dataset()
        tendencies.to_zarr(
            str(tendency_dataset_path.join("ds.zarr")), consolidated=True
        )
        config = get_nudging_config(wrapper, str(tendency_dataset_path.join("ds.zarr")))
    elif configuration == ConfigEnum.microphys_emulation:
        model_path = str(tmpdir_factory.mktemp("model").join("model.tf"))
        model = create_emulation_model()
        model.save(model_path)
        config = get_emulation_config(wrapper, model_path)
    else:
        raise NotImplementedError()

    rundir = tmpdir_factory.mktemp("rundir").join("subdir")
    run_native(config, str(rundir))
    return rundir


@pytest.fixture()
def completed_segment(completed_rundir):
    return completed_rundir.join("artifacts").join("20160801.000000")


def test_fv3run_checksum_restarts(wrapper, completed_segment, regtest):
    """Please do not add more test cases here as this test slows image build time.
    Additional Predictor model types and configurations should be tested against
    the base class in the fv3fit test suite.
    """
    fv_core = completed_segment.join("RESTART").join("fv_core.res.tile1.nc")
    print(fv_core.computehash(), file=regtest)


@pytest.mark.parametrize("path", [LOG_PATH, STATISTICS_PATH, PROFILES_PATH])
def test_fv3run_logs_present(wrapper, completed_segment, path):
    assert completed_segment.join(path).exists()


def test_chunks_present(wrapper, completed_segment):
    assert completed_segment.join(CHUNKS_PATH).exists()


def test_fv3run_diagnostic_outputs_check_variables(wrapper, regtest, completed_rundir):
    """Please do not add more test cases here as this test slows image build time.
    Additional Predictor model types and configurations should be tested against
    the base class in the fv3fit test suite.
    """
    diagnostics = xr.open_zarr(str(completed_rundir.join("diags.zarr")))
    for variable in sorted(diagnostics):
        assert np.sum(np.isnan(diagnostics[variable].values)) == 0
        checksum = vcm.testing.checksum_dataarray(diagnostics[variable])
        print(f"{variable}: " + checksum, file=regtest)


def test_fv3run_diagnostic_outputs_schema(wrapper, regtest, completed_rundir):
    diagnostics = xr.open_zarr(str(completed_rundir.join("diags.zarr")))
    diagnostics.info(regtest)


def test_metrics_valid(wrapper, completed_segment, configuration):
    if configuration == ConfigEnum.nudging:
        pytest.skip()

    path = str(completed_segment.join(STATISTICS_PATH))

    # read python mass conservation info
    with open(path) as f:
        lines = f.readlines()

    assert len(lines) > 0
    for metric in lines:
        obj = json.loads(metric)
        runtime.metrics.validate(obj)


@pytest.mark.xfail
def test_fv3run_python_mass_conserving(wrapper, completed_segment, configuration):
    if configuration == ConfigEnum.nudging:
        pytest.skip()

    path = str(completed_segment.join(STATISTICS_PATH))

    # read python mass conservation info
    with open(path) as f:
        lines = f.readlines()

    for metric in lines:
        obj = json.loads(metric)

        np.testing.assert_allclose(
            obj["storage_of_mass_due_to_python"],
            obj["storage_of_total_water_path_due_to_python"] * 9.81,
            rtol=0.003,
            atol=1e-4 / 86400,
        )


def test_fv3run_vertical_profile_statistics(wrapper, completed_segment, configuration):
    if (
        configuration == ConfigEnum.nudging
        or configuration == ConfigEnum.microphys_emulation
    ):
        # no specific humidity limiter for nudging run
        pytest.skip()
    path = str(completed_segment.join(PROFILES_PATH))
    npz = get_base_config(wrapper)["namelist"]["fv_core_nml"]["npz"]
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        profiles = json.loads(line)
        assert "time" in profiles
        assert len(profiles["specific_humidity_limiter_active_global_sum"]) == npz


def test_fv3run_emulation_zarr_out(wrapper, completed_rundir, configuration, regtest):

    if configuration != ConfigEnum.microphys_emulation:
        pytest.skip()

    emu_state_zarr = xr.open_zarr(str(completed_rundir.join("state_output.zarr")))
    emu_state_zarr.info(regtest)


def test_each_line_of_file_is_json(wrapper, completed_segment):
    with completed_segment.join("logs.txt").open() as f:
        k = 0
        for line in f:
            json.loads(line)
            k += 1
    some_lines_read = k > 0
    assert some_lines_read


def test_fv3run_emulation_nc_out(wrapper, completed_segment, configuration, regtest):

    if configuration != ConfigEnum.microphys_emulation:
        pytest.skip()

    emu_state_nc_path = completed_segment.join("netcdf_output").join(
        "state_20160801.000000_1.nc"
    )
    emu_state_nc = xr.open_dataset(str(emu_state_nc_path))
    emu_state_nc.info(regtest)
