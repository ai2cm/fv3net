import pytest
import os
import xarray as xr
import synth
from loaders.mappers._nudged._nudged import open_nudge_to_fine, open_nudge_to_obs
from loaders.mappers import open_fine_resolution_nudging_hybrid


def save_data_dir(datadir_module, outpath, nudge_schema_path):
    output = os.path.join(datadir_module, outpath)
    schema = synth.load_directory_schema(str(datadir_module.join(nudge_schema_path)))
    synth.write_directory_schema(output, schema)
    return output


@pytest.fixture(scope="module")
def nudge_to_fine_data_dir(datadir_module):
    return save_data_dir(datadir_module, "nudge_to_fine_data", "nudge_to_fine")


@pytest.fixture(scope="module")
def nudge_to_obs_data_dir(datadir_module):
    return save_data_dir(datadir_module, "nudge_to_obs_data", "nudge_to_obs")


@pytest.fixture(scope="module")
def fine_res_zarr(datadir_module):
    return os.path.join(
        save_data_dir(datadir_module, "fine_res", "fine_res"), "fine_res.zarr"
    )


NUDGE_TO_FINE_VARIABLES = [
    "air_temperature",
    "specific_humidity",
    "x_wind",
    "y_wind",
    "pressure_thickness_of_atmospheric_layer",
]


def test_open_nudge_to_fine(nudge_to_fine_data_dir):

    mapper = open_nudge_to_fine(
        nudge_to_fine_data_dir, NUDGE_TO_FINE_VARIABLES, consolidated=False
    )

    key = list(mapper.keys())[0]
    mapper[key]["air_temperature"]
    mapper[key]["specific_humidity"]
    mapper[key]["dQ1"]
    mapper[key]["dQ2"]
    mapper[key]["dQxwind"]
    mapper[key]["dQywind"]
    mapper[key]["pQ1"]
    mapper[key]["pQ2"]
    mapper[key]["pQu"]
    mapper[key]["pQv"]


@pytest.mark.parametrize(
    ["physics_timestep_seconds", "nudge_to_fine_variables"],
    [
        (900.0, NUDGE_TO_FINE_VARIABLES),
        (60.0, NUDGE_TO_FINE_VARIABLES),
        (900.0, NUDGE_TO_FINE_VARIABLES[:2]),
    ],
)
def test_open_nudge_to_fine_subtract_nudging_increment(
    nudge_to_fine_data_dir, physics_timestep_seconds, nudge_to_fine_variables
):

    nudging_variable_state = xr.open_zarr(
        os.path.join(nudge_to_fine_data_dir, "state_after_timestep.zarr"),
        consolidated=False,
    )
    nudge_to_fine_tendencies = xr.open_zarr(
        os.path.join(nudge_to_fine_data_dir, "nudging_tendencies.zarr"),
        consolidated=False,
    )

    mapper = open_nudge_to_fine(
        nudge_to_fine_data_dir,
        nudge_to_fine_variables,
        physics_timestep_seconds=physics_timestep_seconds,
        consolidated=False,
    )

    for nudging_variable in nudge_to_fine_variables:
        before_nudging_variable_state = (
            nudging_variable_state[nudging_variable]
            - nudge_to_fine_tendencies[f"{nudging_variable}_tendency_due_to_nudging"]
            * physics_timestep_seconds
        ).isel(time=0)
        key = sorted(list(mapper.keys()))[0]
        xr.testing.assert_allclose(
            mapper[key][nudging_variable], before_nudging_variable_state
        )


NUDGE_TO_OBS_VARIABLES = {
    "air_temperature": "dQ1",
    "specific_humidity": "dQ2",
    "eastward_wind": "dQu",
    "northward_wind": "dQv",
}


def test_open_nudge_to_obs(nudge_to_obs_data_dir):

    mapper = open_nudge_to_obs(
        nudge_to_obs_data_dir,
        nudging_tendency_variables=NUDGE_TO_OBS_VARIABLES,
        consolidated=False,
    )
    key = list(mapper.keys())[0]
    mapper[key]["air_temperature"]
    mapper[key]["specific_humidity"]
    mapper[key]["dQ1"]
    mapper[key]["dQ2"]
    mapper[key]["dQu"]
    mapper[key]["dQv"]
    mapper[key]["pQ1"]
    mapper[key]["pQ2"]
    mapper[key]["pQu"]
    mapper[key]["pQv"]


@pytest.mark.parametrize(
    ["physics_timestep_seconds", "nudge_to_obs_variables"],
    [
        (900.0, NUDGE_TO_OBS_VARIABLES),
        (60.0, NUDGE_TO_OBS_VARIABLES),
        (900.0, {"air_temperature": "dQ1"}),
    ],
)
def test_open_nudge_to_obs_subtract_nudging_increment(
    nudge_to_obs_data_dir, physics_timestep_seconds, nudge_to_obs_variables
):

    nudging_variable_state = xr.open_zarr(
        os.path.join(nudge_to_obs_data_dir, "state_after_timestep.zarr"),
        consolidated=False,
    )
    nudge_to_obs_tendencies = xr.open_zarr(
        os.path.join(nudge_to_obs_data_dir, "nudging_tendencies.zarr"),
        consolidated=False,
    ).rename(
        {
            "t_dt_nudge": "dQ1",
            "q_dt_nudge": "dQ2",
            "u_dt_nudge": "dQu",
            "v_dt_nudge": "dQv",
            "grid_xt": "x",
            "grid_yt": "y",
            "pfull": "z",
        }
    )

    mapper = open_nudge_to_obs(
        nudge_to_obs_data_dir,
        nudge_to_obs_variables,
        physics_timestep_seconds=physics_timestep_seconds,
        consolidated=False,
    )

    for nudged_variable_name, nudging_tendency_name in nudge_to_obs_variables.items():
        before_nudging_variable_state = (
            nudging_variable_state[nudged_variable_name]
            - nudge_to_obs_tendencies[nudging_tendency_name] * physics_timestep_seconds
        ).isel(time=0)
        key = sorted(list(mapper.keys()))[0]
        xr.testing.assert_allclose(
            mapper[key][nudged_variable_name], before_nudging_variable_state
        )


@pytest.mark.parametrize(
    ["nudge_to_obs_variables"],
    [(NUDGE_TO_OBS_VARIABLES,), ({"air_temperature": "dQ1"},)],
)
def test_open_nudge_to_obs_subtract_nudging_tendency(
    nudge_to_obs_data_dir, nudge_to_obs_variables
):

    physics_tendencies = xr.open_zarr(
        os.path.join(nudge_to_obs_data_dir, "physics_tendencies.zarr"),
        consolidated=False,
    ).rename(
        {
            "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
            "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
            "tendency_of_eastward_wind_due_to_fv3_physics": "pQu",
            "tendency_of_northward_wind_due_to_fv3_physics": "pQv",
        }
    )

    nudge_to_obs_tendencies = xr.open_zarr(
        os.path.join(nudge_to_obs_data_dir, "nudging_tendencies.zarr"),
        consolidated=False,
    ).rename(
        {
            "t_dt_nudge": "dQ1",
            "q_dt_nudge": "dQ2",
            "u_dt_nudge": "dQu",
            "v_dt_nudge": "dQv",
            "grid_xt": "x",
            "grid_yt": "y",
            "pfull": "z",
        }
    )

    mapper = open_nudge_to_obs(
        nudge_to_obs_data_dir, nudge_to_obs_variables, consolidated=False,
    )

    physics_nudging_mapping = {"dQ1": "pQ1", "dQ2": "pQ2", "dQu": "pQu", "dQv": "pQv"}

    for nudging_tendency_name, physics_tendency_name in physics_nudging_mapping.items():
        physics_tendency = (
            physics_tendencies[physics_tendency_name]
            - nudge_to_obs_tendencies[nudging_tendency_name]
        ).isel(time=0)
        key = sorted(list(mapper.keys()))[0]
        xr.testing.assert_allclose(mapper[key][physics_tendency_name], physics_tendency)


timestep1 = "20160801.000730"
timestep1_end = "20160801.001500"
timestep2 = "20160801.002230"
times_centered_str = [timestep1, timestep2]


@pytest.fixture
def fine_url(tmpdir):
    fine_url = str(tmpdir.mkdir("fine_res"))
    synth.generate_fine_res(fine_url, times_centered_str)
    return fine_url


def test_open_fine_resolution_nudging_hybrid(nudge_to_fine_data_dir, fine_res_zarr):
    data = open_fine_resolution_nudging_hybrid(
        None, fine_url=fine_res_zarr, nudge_url=nudge_to_fine_data_dir,
    )
    data[timestep1_end]
