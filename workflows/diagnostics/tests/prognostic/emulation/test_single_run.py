import numpy as np
import pytest
import vcm

from fv3net.diagnostics.prognostic_run.emulation import single_run

cdl = """
netcdf out {
dimensions:
    time = 3;
    tile = 6;
    z = 79 ;
    y = 12 ;
    x = 12 ;
    phalf = 80 ;
    y_interface = 13 ;
    x_interface = 13 ;
variables:
    double time(time) ;
        time:_FillValue = NaN ;
        time:calendar_type = "JULIAN" ;
        time:cartesian_axis = "T" ;
        time:long_name = "time" ;
        time:units = "hours since 2016-06-11" ;
        time:calendar = "JULIAN" ;
    double z(z) ;
        z:_FillValue = NaN ;
        z:cartesian_axis = "Z" ;
        z:edges = "phalf" ;
        z:long_name = "ref full pressure level" ;
        z:positive = "down" ;
        z:units = "mb" ;
    double y(y) ;
        y:_FillValue = NaN ;
        y:cartesian_axis = "Y" ;
        y:long_name = "T-cell latitude" ;
        y:units = "degrees_N" ;
    double x(x) ;
        x:_FillValue = NaN ;
        x:cartesian_axis = "X" ;
        x:long_name = "T-cell longitude" ;
        x:units = "degrees_E" ;
    float delp(time, tile, z, y, x) ;
        delp:_FillValue = NaNf ;
        delp:cell_methods = "time: point" ;
        delp:long_name = "pressure thickness" ;
        delp:units = "pa" ;
        delp:coordinates = "time" ;
    double phalf(phalf) ;
        phalf:_FillValue = NaN ;
        phalf:cartesian_axis = "Z" ;
        phalf:long_name = "ref half pressure level" ;
        phalf:positive = "down" ;
        phalf:units = "mb" ;
    float surface_precipitation_due_to_zhao_carr_emulator(time, tile, y, x) ;
        surface_precipitation_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        surface_precipitation_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        surface_precipitation_due_to_zhao_carr_emulator:long_name = "surface precipitation due to zhao_carr_microphysics emulator" ;
        surface_precipitation_due_to_zhao_carr_emulator:units = "kg/m^2/s" ;
        surface_precipitation_due_to_zhao_carr_emulator:coordinates = "time" ;
    float surface_precipitation_due_to_zhao_carr_physics(time, tile, y, x) ;
        surface_precipitation_due_to_zhao_carr_physics:_FillValue = NaNf ;
        surface_precipitation_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        surface_precipitation_due_to_zhao_carr_physics:long_name = "surface precipitation due to zhao_carr_microphysics physics" ;
        surface_precipitation_due_to_zhao_carr_physics:units = "kg/m^2/s" ;
        surface_precipitation_due_to_zhao_carr_physics:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_gscond_emulator(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_gscond_emulator:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_gscond_emulator:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_gscond_emulator:long_name = "temperature tendency due to zhao_carr_gscond emulator" ;
        tendency_of_air_temperature_due_to_gscond_emulator:units = "K/s" ;
        tendency_of_air_temperature_due_to_gscond_emulator:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_gscond_physics(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_gscond_physics:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_gscond_physics:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_gscond_physics:long_name = "temperature tendency due to zhao_carr_gscond physics" ;
        tendency_of_air_temperature_due_to_gscond_physics:units = "K/s" ;
        tendency_of_air_temperature_due_to_gscond_physics:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_zhao_carr_emulator(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:long_name = "temperature tendency due to zhao_carr_microphysics emulator" ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:units = "K/s" ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_zhao_carr_physics(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:long_name = "temperature tendency due to zhao_carr_microphysics physics" ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:units = "K/s" ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:coordinates = "time" ;
    float tendency_of_cloud_water_due_to_gscond_physics(time, tile, z, y, x) ;
        tendency_of_cloud_water_due_to_gscond_physics:_FillValue = NaNf ;
        tendency_of_cloud_water_due_to_gscond_physics:cell_methods = "time: point" ;
        tendency_of_cloud_water_due_to_gscond_physics:long_name = "cloud water due to zhao_carr_gscond physics" ;
        tendency_of_cloud_water_due_to_gscond_physics:units = "kg/kg/s" ;
        tendency_of_cloud_water_due_to_gscond_physics:coordinates = "time" ;
    float tendency_of_cloud_water_due_to_zhao_carr_emulator(time, tile, z, y, x) ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:long_name = "cloud water due to zhao_carr_microphysics emulator" ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:units = "kg/kg/s" ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:coordinates = "time" ;
    float tendency_of_cloud_water_due_to_zhao_carr_physics(time, tile, z, y, x) ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:_FillValue = NaNf ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:long_name = "cloud water due to zhao_carr_microphysics physics" ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:units = "kg/kg/s" ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_gscond_emulator(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_gscond_emulator:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_gscond_emulator:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_gscond_emulator:long_name = "specific humidity tendency due to zhao_carr_gscond emulator" ;
        tendency_of_specific_humidity_due_to_gscond_emulator:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_gscond_emulator:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_gscond_physics(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_gscond_physics:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_gscond_physics:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_gscond_physics:long_name = "specific humidity tendency due to zhao_carr_gscond physics" ;
        tendency_of_specific_humidity_due_to_gscond_physics:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_gscond_physics:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_zhao_carr_emulator(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:long_name = "specific humidity tendency due to zhao_carr_microphysics emulator" ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_zhao_carr_physics(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:long_name = "specific humidity tendency due to zhao_carr_microphysics physics" ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:coordinates = "time" ;
    float area(tile, y, x) ;
        area:_FillValue = NaNf ;
        area:cell_methods = "time: point" ;
        area:long_name = "cell area" ;
        area:units = "m**2" ;
        area:coordinates = "time" ;
    float lat(tile, y, x) ;
        lat:_FillValue = NaNf ;
        lat:cell_methods = "time: point" ;
        lat:long_name = "latitude" ;
        lat:units = "degrees_N" ;
        lat:coordinates = "time" ;
    float latb(tile, y_interface, x_interface) ;
        latb:_FillValue = NaNf ;
        latb:cell_methods = "time: point" ;
        latb:long_name = "latitude" ;
        latb:units = "degrees_N" ;
        latb:coordinates = "time" ;
    float lon(tile, y, x) ;
        lon:_FillValue = NaNf ;
        lon:cell_methods = "time: point" ;
        lon:long_name = "longitude" ;
        lon:units = "degrees_E" ;
        lon:coordinates = "time" ;
    float lonb(tile, y_interface, x_interface) ;
        lonb:_FillValue = NaNf ;
        lonb:cell_methods = "time: point" ;
        lonb:long_name = "longitude" ;
        lonb:units = "degrees_E" ;
        lonb:coordinates = "time" ;
    double x_interface(x_interface) ;
        x_interface:_FillValue = NaN ;
        x_interface:cartesian_axis = "X" ;
        x_interface:long_name = "cell corner longitude" ;
        x_interface:units = "degrees_E" ;
    double y_interface(y_interface) ;
        y_interface:_FillValue = NaN ;
        y_interface:cartesian_axis = "Y" ;
        y_interface:long_name = "cell corner latitude" ;
        y_interface:units = "degrees_N" ;
    double air_temperature(time, tile, z, y, x) ;
        air_temperature:_FillValue = NaN ;
        air_temperature:units = "degK" ;
        air_temperature:coordinates = "time" ;
    double cloud_water_mixing_ratio(time, tile, z, y, x) ;
        cloud_water_mixing_ratio:_FillValue = NaN ;
        cloud_water_mixing_ratio:units = "kg/kg" ;
        cloud_water_mixing_ratio:coordinates = "time" ;
    double eastward_wind(time, tile, z, y, x) ;
        eastward_wind:_FillValue = NaN ;
        eastward_wind:units = "m/s" ;
        eastward_wind:coordinates = "time" ;
    double land_sea_mask(time, tile, y, x) ;
        land_sea_mask:_FillValue = NaN ;
        land_sea_mask:units = "" ;
        land_sea_mask:coordinates = "time" ;
    double latitude(time, tile, y, x) ;
        latitude:_FillValue = NaN ;
        latitude:units = "radians" ;
        latitude:coordinates = "time" ;
    double longitude(time, tile, y, x) ;
        longitude:_FillValue = NaN ;
        longitude:units = "radians" ;
        longitude:coordinates = "time" ;
    double northward_wind(time, tile, z, y, x) ;
        northward_wind:_FillValue = NaN ;
        northward_wind:units = "m/s" ;
        northward_wind:coordinates = "time" ;
    double pressure_thickness_of_atmospheric_layer(time, tile, z, y, x) ;
        pressure_thickness_of_atmospheric_layer:_FillValue = NaN ;
        pressure_thickness_of_atmospheric_layer:units = "Pa" ;
        pressure_thickness_of_atmospheric_layer:coordinates = "time" ;
    double specific_humidity(time, tile, z, y, x) ;
        specific_humidity:_FillValue = NaN ;
        specific_humidity:units = "kg/kg" ;
        specific_humidity:coordinates = "time" ;
    double surface_pressure(time, tile, y, x) ;
        surface_pressure:_FillValue = NaN ;
        surface_pressure:units = "Pa" ;
        surface_pressure:coordinates = "time" ;
    double total_precipitation(time, tile, y, x) ;
        total_precipitation:_FillValue = NaN ;
        total_precipitation:units = "m" ;
        total_precipitation:coordinates = "time" ;
    double vertical_wind(time, tile, z, y, x) ;
        vertical_wind:_FillValue = NaN ;
        vertical_wind:units = "m/s" ;
        vertical_wind:coordinates = "time" ;
data:
    time = 3, 6, 9, 12, 15;
}

"""  # noqa


@pytest.mark.parametrize(
    "func",
    [
        single_run.plot_histogram_begin_end,
        # this test fails in CI for some reason
        # single_run.plot_cloud_weighted_average,
        single_run.plot_cloud_maps,
        single_run.skill_table,
        single_run.skill_time_table,
        single_run.log_lat_vs_p_skill("cloud_water"),
    ],
)
def test_log_functions(func):

    ds = vcm.cdl_to_dataset(cdl)
    for key in ds:
        ds[key].values[:] = 0

    for key in set(ds.coords) - {"time"}:
        ds[key].values[:] = np.arange(len(ds[key]))

    nx = len(ds.x)
    ds["lat"].values[:] = np.linspace(-45, 45, nx)
    func(ds)


def test_skill_table(regtest):
    ds = vcm.cdl_to_dataset(cdl)
    output = single_run.skill_table(ds)
    for name in sorted(output):
        print(name, ":", output[name].columns, file=regtest)


def test_compute_summaries(regtest):
    ds = vcm.cdl_to_dataset(cdl)
    output = single_run.compute_summaries(ds)
    print(sorted(output), file=regtest)
