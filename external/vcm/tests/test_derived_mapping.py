import cftime
import numpy as np
import pytest
from typing import Mapping
import xarray as xr

from vcm import DerivedMapping

nt, nx, ny = 3, 2, 1
ds = xr.Dataset(
    {
        "lat": xr.DataArray(np.random.rand(ny, nx), dims=["y", "x"]),
        "lon": xr.DataArray(np.random.rand(ny, nx), dims=["y", "x"]),
        "T": xr.DataArray(
            np.random.rand(ny, nx, nt),
            dims=["y", "x", "time"],
            coords={"time": [cftime.DatetimeJulian(2016, 1, 1) for n in range(nt)]},
        ),
        "q": xr.DataArray(
            np.random.rand(ny, nx, nt),
            dims=["y", "x", "time"],
            coords={"time": [cftime.DatetimeJulian(2016, 1, 1) for n in range(nt)]},
        ),
        "dQu": xr.DataArray(
            np.ones((ny, nx, nt)),
            dims=["y", "x", "time"],
            coords={"time": [cftime.DatetimeJulian(2016, 1, 1) for n in range(nt)]},
        ),
    }
)


@pytest.mark.parametrize(
    "dQu, dQv, eastward, northward, projection",
    [
        pytest.param(1.0, 0.0, 1.0, 0.0, 1.0, id="parallel, east wind"),
        pytest.param(1.0, 0.0, -1.0, 0.0, -1.0, id="antiparallel, east wind"),
        pytest.param(1.0, 1.0, 1.0, 1.0, 1, id="parallel, 45 deg NE wind"),
        pytest.param(1.0, 0.0, 1.0, 1.0, np.sqrt(2), id="45 deg CCW"),
        pytest.param(-1.0, 0.0, 1.0, 1.0, -np.sqrt(2), id="135 deg CCW"),
    ],
)
def test_horizontal_wind_tendency_parallel_to_horizontal_wind(
    dQu, dQv, eastward, northward, projection
):
    data = xr.Dataset(
        {
            "dQu": xr.DataArray([dQu], dims=["x"]),
            "dQv": xr.DataArray([dQv], dims=["x"]),
            "eastward_wind": xr.DataArray([eastward], dims=["x"]),
            "northward_wind": xr.DataArray([northward], dims=["x"]),
        }
    )
    derived_mapping = DerivedMapping(data)
    assert pytest.approx(
        derived_mapping[
            "horizontal_wind_tendency_parallel_to_horizontal_wind"
        ].values.item(),
        projection,
    )


def _dataset_with_d_grid_winds_and_tendencies():
    rotation = {
        var: xr.DataArray(np.zeros((ny, nx)), dims=["y", "x"])
        for var in [
            "eastward_wind_u_coeff",
            "eastward_wind_v_coeff",
            "northward_wind_u_coeff",
            "northward_wind_v_coeff",
        ]
    }
    data = xr.Dataset(
        {
            "dQxwind": xr.DataArray(np.ones((ny + 1, nx)), dims=["y_interface", "x"]),
            "dQywind": xr.DataArray(np.ones((ny, nx + 1)), dims=["y", "x_interface"]),
            "x_wind": xr.DataArray(np.ones((ny + 1, nx)), dims=["y_interface", "x"]),
            "y_wind": xr.DataArray(np.ones((ny, nx + 1)), dims=["y", "x_interface"]),
            **rotation,
        }
    )
    return data


@pytest.mark.parametrize("variable", ["dQu", "dQv", "eastward_wind", "northward_wind"])
def test_rotated_winds(variable):
    data = _dataset_with_d_grid_winds_and_tendencies()
    derived_mapping = DerivedMapping(data)
    np.testing.assert_array_almost_equal(0.0, derived_mapping[variable])


def test_wind_tendency_nonderived():
    # dQu/dQv already exist in data
    derived_mapping = DerivedMapping(ds)
    dQu = derived_mapping["dQu"]
    np.testing.assert_array_almost_equal(dQu, 1.0)


def test_DerivedMapping():
    derived_state = DerivedMapping(ds)
    assert isinstance(derived_state["T"], xr.DataArray)


def test_DerivedMapping_cos_zenith():
    derived_state = DerivedMapping(ds)
    output = derived_state["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


def test_DerivedMapping__data_arrays():
    derived_state = DerivedMapping(ds)
    keys = ["T", "q"]
    data_arrays = derived_state._data_arrays(keys)
    assert isinstance(data_arrays, Mapping)
    assert set(keys) == set(data_arrays.keys())


def test_DerivedMapping_dataset():
    derived_state = DerivedMapping(ds)
    keys = ["T", "q", "cos_zenith_angle"]
    ds_derived_state = derived_state.dataset(keys)
    assert isinstance(ds_derived_state, xr.Dataset)
    for existing_var in ["T", "q"]:
        np.testing.assert_array_almost_equal(
            ds_derived_state[existing_var], ds[existing_var]
        )


def test_DerivedMapping_unregistered():
    derived_state = DerivedMapping(ds)
    with pytest.raises(KeyError):
        derived_state["latent_heat_flux"]


def test_net_downward_shortwave_sfc_flux_derived():
    ds = xr.Dataset(
        {
            "surface_diffused_shortwave_albedo": xr.DataArray(
                [0, 0.5, 1.0], dims=["x"]
            ),
            "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface": (
                xr.DataArray([1.0, 1.0, 1.0], dims=["x"])
            ),
        }
    )
    derived_state = DerivedMapping(ds)
    derived_net_sw = derived_state["net_shortwave_sfc_flux_derived"]
    np.testing.assert_array_almost_equal(derived_net_sw, [1.0, 0.5, 0.0])


def test_required_inputs():
    @DerivedMapping.register("test_derived_var", required_inputs=["required_input"])
    def test_derived_var(self):
        return None

    assert set(DerivedMapping.REQUIRED_INPUTS["test_derived_var"]) == {"required_input"}


ds_sfc = xr.Dataset({"land_sea_mask": xr.DataArray([0, 1, 2], dims=["x"])})


def test_is_sea():
    derived_state = DerivedMapping(ds_sfc)
    np.testing.assert_array_almost_equal(derived_state["is_sea"], [1.0, 0.0, 0.0])


def test_is_land():
    derived_state = DerivedMapping(ds_sfc)
    np.testing.assert_array_almost_equal(derived_state["is_land"], [0.0, 1.0, 0.0])


def test_is_sea_ice():
    derived_state = DerivedMapping(ds_sfc)
    np.testing.assert_array_almost_equal(derived_state["is_sea_ice"], [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "dependency_map, derived_vars, reqs",
    [
        ({"c": ["d"]}, ["a"], []),
        ({"a": ["b"], "c": ["d"]}, ["a"], ["b"]),
        ({"a": ["b"], "c": ["d"]}, ["a", "c"], ["b", "d"]),
        ({"a": ["b"], "b": ["c"], "c": ["d"]}, ["a"], ["b", "c", "d"]),
        ({"a": ["b"], "c": ["d"], "b": ["e"]}, ["a", "c"], ["b", "d", "e"]),
    ],
)
def test_find_all_required_inputs(dependency_map, derived_vars, reqs):
    for var, dependencies in dependency_map.items():

        @DerivedMapping.register(var, required_inputs=dependencies)
        def var(self):
            return None

    required_inputs = DerivedMapping.find_all_required_inputs(derived_vars)
    assert set(required_inputs) == set(reqs)
    assert len(required_inputs) == len(reqs)


def get_microphys_data(varname, diff):

    ds = xr.Dataset(
        {
            f"{varname}_input": xr.DataArray(
                np.ones((10, 79)), dims=["sample", "z"]
            ),
            f"{varname}_output": xr.DataArray(
                np.ones((10, 79)), dims=["sample", "z"]
            ) + diff,
        }
    )

    return ds


@pytest.mark.parametrize(
    "varname",
    ["air_temperature", "specific_humidity", "cloud_water_mixing_ratio"]
)
def test_microphysics_tendencies(varname):

    timestep_sec = 2
    ds = get_microphys_data(varname, timestep_sec)
    derived = DerivedMapping(ds, microphys_timestep_sec=timestep_sec)

    tend = derived[f"tendency_of_{varname}_due_to_microphysics"]
    np.testing.assert_array_equal(np.array(tend), np.ones_like(tend))
