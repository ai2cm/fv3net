from typing import Union
import xarray
from typing_extensions import Protocol
from pathlib import Path

import vcm.convenience
import vcm.cubedsphere
import vcm.fv3.metadata


def mse(truth, prediction: Union[xarray.DataArray, float], area):
    units = "(" + truth.units + ")^2"
    error = ((truth - prediction) ** 2 * area).sum(area.dims) / area.sum(area.dims)
    return error.assign_attrs(units=units)


class Data(Protocol):
    PRESsfc: xarray.DataArray
    PRMSL: xarray.DataArray
    PWAT: xarray.DataArray
    RH1000: xarray.DataArray
    RH500: xarray.DataArray
    RH700: xarray.DataArray
    RH850: xarray.DataArray
    RH925: xarray.DataArray
    SOILM: xarray.DataArray
    TMP200: xarray.DataArray
    TMP500: xarray.DataArray
    TMP500_300: xarray.DataArray
    TMP850: xarray.DataArray
    TMPlowest: xarray.DataArray
    UGRD200: xarray.DataArray
    UGRD50: xarray.DataArray
    UGRD500: xarray.DataArray
    UGRD850: xarray.DataArray
    UGRDlowest: xarray.DataArray
    VGRD200: xarray.DataArray
    VGRD50: xarray.DataArray
    VGRD500: xarray.DataArray
    VGRD850: xarray.DataArray
    VGRDlowest: xarray.DataArray
    VIL: xarray.DataArray
    VORT200: xarray.DataArray
    VORT500: xarray.DataArray
    VORT850: xarray.DataArray
    area: xarray.DataArray
    convective_precipitation_diagnostic: xarray.DataArray
    h200: xarray.DataArray
    h500: xarray.DataArray
    h850: xarray.DataArray
    iw: xarray.DataArray
    kinetic_energy: xarray.DataArray
    lat: xarray.DataArray
    latb: xarray.DataArray
    latent_heat_flux_diagnostic: xarray.DataArray
    lon: xarray.DataArray
    lonb: xarray.DataArray
    q1000: xarray.DataArray
    q500: xarray.DataArray
    q700: xarray.DataArray
    q850: xarray.DataArray
    q925: xarray.DataArray
    sensible_heat_flux_diagnostics: xarray.DataArray
    total_energy: xarray.DataArray
    total_precipitation_diagnostic: xarray.DataArray
    w200: xarray.DataArray
    w500: xarray.DataArray
    w850: xarray.DataArray
    storage_of_air_temperature_path_due_to_emulator: xarray.DataArray
    storage_of_air_temperature_path_due_to_fv3_physics: xarray.DataArray
    storage_of_eastward_wind_path_due_to_emulator: xarray.DataArray
    storage_of_eastward_wind_path_due_to_fv3_physics: xarray.DataArray
    storage_of_northward_wind_path_due_to_emulator: xarray.DataArray
    storage_of_northward_wind_path_due_to_fv3_physics: xarray.DataArray
    storage_of_specific_humidity_path_due_to_emulator: xarray.DataArray
    storage_of_specific_humidity_path_due_to_fv3_physics: xarray.DataArray
    water_vapor_path: xarray.DataArray
    tendency_of_cloud_water_mixing_ratio_due_to_emulator: xarray.DataArray
    tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics: xarray.DataArray
    tendency_of_air_temperature_due_to_emulator: xarray.DataArray
    tendency_of_air_temperature_due_to_fv3_physics: xarray.DataArray
    tendency_of_eastward_wind_due_to_emulator: xarray.DataArray
    tendency_of_eastward_wind_due_to_fv3_physics: xarray.DataArray
    tendency_of_northward_wind_due_to_emulator: xarray.DataArray
    tendency_of_northward_wind_due_to_fv3_physics: xarray.DataArray
    tendency_of_specific_humidity_due_to_emulator: xarray.DataArray
    tendency_of_specific_humidity_due_to_fv3_physics: xarray.DataArray


def compute_metrics(ds: Data) -> xarray.Dataset:
    return xarray.Dataset(
        dict(
            t_null_error=mse(
                ds.tendency_of_air_temperature_due_to_fv3_physics, 0, ds.area
            ),
            t_error=mse(
                ds.tendency_of_air_temperature_due_to_fv3_physics,
                ds.tendency_of_air_temperature_due_to_emulator,
                ds.area,
            ),
            q_null_error=mse(
                ds.tendency_of_specific_humidity_due_to_fv3_physics, 0.0, ds.area
            ),
            q_error=mse(
                ds.tendency_of_specific_humidity_due_to_fv3_physics,
                ds.tendency_of_specific_humidity_due_to_emulator,
                ds.area,
            ),
            u_null_error=mse(
                ds.tendency_of_eastward_wind_due_to_fv3_physics, 0.0, ds.area
            ),
            u_error=mse(
                ds.tendency_of_eastward_wind_due_to_fv3_physics,
                ds.tendency_of_eastward_wind_due_to_emulator,
                ds.area,
            ),
            v_null_error=mse(
                ds.tendency_of_northward_wind_due_to_fv3_physics, 0.0, ds.area
            ),
            v_error=mse(
                ds.tendency_of_northward_wind_due_to_fv3_physics,
                ds.tendency_of_northward_wind_due_to_emulator,
                ds.area,
            ),
            qc_error=mse(
                ds.tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics,
                ds.tendency_of_cloud_water_mixing_ratio_due_to_emulator,
                ds.area,
            ),
            qc_null_error=mse(
                ds.tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics,
                0.0,
                ds.area,
            ),
        )
    )


def open_run(path: Path) -> Data:
    fv_diags = vcm.fv3.metadata.gfdl_to_standard(
        xarray.open_zarr(path / "sfc_dt_atmos.zarr")
    )
    grid = vcm.catalog.catalog["grid/c48"].to_dask()
    diags_2d = xarray.open_zarr(path / "diags.zarr")
    diags_3d = xarray.open_zarr(path / "diags_3d.zarr")
    return xarray.merge([grid, fv_diags, diags_2d, diags_3d], compat="override")
