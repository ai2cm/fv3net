from .manager import DiagnosticFileConfig, FortranFileConfig

ml_diagnostics = DiagnosticFileConfig(
    name="diags.zarr",
    chunks={"time": 96},
    variables=[
        "net_moistening",
        "net_moistening_diagnostic",
        "net_heating",
        "net_heating_diagnostic",
        "total_precipitation_rate",
        "water_vapor_path",
        "physics_precip",
        "column_integrated_dQu",
        "column_integrated_dQu_diagnostic",
        "column_integrated_dQv",
        "column_integrated_dQv_diagnostic",
    ],
)
nudging_diagnostics_2d = DiagnosticFileConfig(
    name="diags.zarr",
    chunks={"time": 96},
    variables=[
        "net_moistening_due_to_nudging",
        "net_heating_due_to_nudging",
        "net_mass_tendency_due_to_nudging",
        "total_precipitation_rate",
        "water_vapor_path",
        "physics_precip",
    ],
)
nudging_tendencies = DiagnosticFileConfig(
    name="nudging_tendencies.zarr", chunks={"time": 8}, variables=[]
)
physics_tendencies = DiagnosticFileConfig(
    name="physics_tendencies.zarr",
    chunks={"time": 8},
    variables=[
        "tendency_of_air_temperature_due_to_fv3_physics",
        "tendency_of_specific_humidity_due_to_fv3_physics",
        "tendency_of_eastward_wind_due_to_fv3_physics",
        "tendency_of_northward_wind_due_to_fv3_physics",
    ],
)
baseline_diagnostics = DiagnosticFileConfig(
    name="diags.zarr",
    chunks={"time": 96},
    variables=["water_vapor_path", "physics_precip"],
)
state_after_timestep = DiagnosticFileConfig(
    name="state_after_timestep.zarr",
    chunks={"time": 8},
    variables=[
        "x_wind",
        "y_wind",
        "eastward_wind",
        "northward_wind",
        "vertical_wind",
        "air_temperature",
        "specific_humidity",
        "pressure_thickness_of_atmospheric_layer",
        "vertical_thickness_of_atmospheric_layer",
        "land_sea_mask",
        "surface_temperature",
        "surface_geopotential",
        "sensible_heat_flux",
        "latent_heat_flux",
        "total_precipitation",
        "surface_precipitation_rate",
        "total_soil_moisture",
        "total_sky_downward_shortwave_flux_at_surface",
        "total_sky_upward_shortwave_flux_at_surface",
        "total_sky_downward_longwave_flux_at_surface",
        "total_sky_upward_longwave_flux_at_surface",
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
        "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
        "total_sky_upward_longwave_flux_at_top_of_atmosphere",
        "clear_sky_downward_shortwave_flux_at_surface",
        "clear_sky_upward_shortwave_flux_at_surface",
        "clear_sky_downward_longwave_flux_at_surface",
        "clear_sky_upward_longwave_flux_at_surface",
        "clear_sky_upward_shortwave_flux_at_top_of_atmosphere",
        "clear_sky_upward_longwave_flux_at_top_of_atmosphere",
        "latitude",
        "longitude",
    ],
)
reference_state = DiagnosticFileConfig(
    name="reference_state.zarr", chunks={"time": 8}, variables=[]
)
sfc_dt_atmos = FortranFileConfig(name="sfc_dt_atmos.zarr", chunks={"time": 96})
atmos_dt_atmos = FortranFileConfig(name="atmos_dt_atmos.zarr", chunks={"time": 96})
atmos_8xdaily = FortranFileConfig(name="atmos_8xdaily.zarr", chunks={"time": 8})
nudging_tendencies_fortran = FortranFileConfig(
    name="nudging_tendencies.zarr", chunks={"time": 8}
)
