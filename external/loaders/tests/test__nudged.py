import pytest
import os
import xarray as xr
import synth
from loaders import TIME_NAME
from loaders.mappers._nudged._nudged import open_nudge_to_fine, open_nudge_to_obs

NTIMES = 2


@pytest.fixture(scope="module")
def state_after_timestep(datadir_module):
    nudge_data_schema = datadir_module.join(f"state_after_timestep.json")
    with open(nudge_data_schema) as f:
        schema = synth.load(f)

    data = synth.generate(schema)

    return data.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudge_to_fine_tendencies(datadir_module):
    nudge_to_fine_tendencies_schema = datadir_module.join(
        f"nudge_to_fine_tendencies.json"
    )
    with open(nudge_to_fine_tendencies_schema) as f:
        schema = synth.load(f)

    nudge_to_fine_tendencies = synth.generate(schema)

    return nudge_to_fine_tendencies.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudge_to_obs_tendencies(datadir_module):
    nudge_to_fine_tendencies_schema = datadir_module.join(
        f"nudge_to_obs_tendencies.json"
    )
    with open(nudge_to_fine_tendencies_schema) as f:
        schema = synth.load(f)

    nudge_to_obs_tendencies = synth.generate(schema)

    return nudge_to_obs_tendencies.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def physics_tendencies(datadir_module):
    physics_tendencies_schema = datadir_module.join(f"physics_tendencies.json")
    with open(physics_tendencies_schema) as f:
        schema = synth.load(f)

    physics_tendencies = synth.generate(schema)

    return physics_tendencies.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudge_to_fine_data_dir(
    datadir_module, state_after_timestep, physics_tendencies, nudge_to_fine_tendencies,
):
    all_data = {"physics_tendencies": physics_tendencies}
    all_data.update({"nudge_to_fine_tendencies": nudge_to_fine_tendencies})
    all_data.update({"state_after_timestep": state_after_timestep})

    nudge_to_fine_path = os.path.join(datadir_module, "nudge_to_fine")

    for filestem, ds in all_data.items():
        filepath = os.path.join(nudge_to_fine_path, f"{filestem}.zarr")
        ds.to_zarr(filepath)

    return str(nudge_to_fine_path)


@pytest.fixture(scope="module")
def nudge_to_obs_data_dir(
    datadir_module, state_after_timestep, physics_tendencies, nudge_to_obs_tendencies,
):
    nudge_to_obs_tendencies = nudge_to_obs_tendencies.assign_coords(
        {"time": state_after_timestep.time}
    )
    all_data = {"physics_tendencies": physics_tendencies}
    all_data.update({"nudge_to_obs_tendencies": nudge_to_obs_tendencies})
    all_data.update({"state_after_timestep": state_after_timestep})

    nudge_to_obs_path = os.path.join(datadir_module, "nudge_to_obs")

    for filestem, ds in all_data.items():
        filepath = os.path.join(nudge_to_obs_path, f"{filestem}.zarr")
        ds.to_zarr(filepath)

    return str(nudge_to_obs_path)


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
        os.path.join(nudge_to_fine_data_dir, "nudge_to_fine_tendencies.zarr"),
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
        os.path.join(nudge_to_obs_data_dir, "nudge_to_obs_tendencies.zarr"),
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
        os.path.join(nudge_to_obs_data_dir, "nudge_to_obs_tendencies.zarr"),
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
