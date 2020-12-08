from runtime import DiagnosticFile

DEFAULT_TIMES = {"kind": "interval", "frequency": 900}
ml_diagnostics = DiagnosticFile(
    name="diags.zarr",
    variables=[
        "net_moistening",
        "net_moistening_diagnostic",
        "net_heating",
        "net_heating_diagnostic",
        "water_vapor_path",
        "physics_precip",
        "column_integrated_dQu",
        "column_integrated_dQu_diagnostic",
        "column_integrated_dQv",
        "column_integrated_dQv_diagnostic",
    ],
    times=DEFAULT_TIMES,
)
nudge_to_fine_diagnostics_2d = DiagnosticFile(
    name="diags.zarr",
    variables=[
        "net_moistening_due_to_nudging",
        "net_heating_due_to_nudging",
        "net_mass_tendency_due_to_nudging",
        "column_integrated_eastward_wind_tendency_due_to_nudging",
        "column_integrated_northward_wind_tendency_due_to_nudging",
        "water_vapor_path",
        "physics_precip",
    ],
    times=DEFAULT_TIMES,
)
nudge_to_fine_tendencies = DiagnosticFile(
    name="nudge_to_fine_tendencies.zarr", variables=[], times=DEFAULT_TIMES
)
physics_tendencies = DiagnosticFile(
    name="physics_tendencies.zarr",
    variables=[
        "tendency_of_air_temperature_due_to_fv3_physics",
        "tendency_of_specific_humidity_due_to_fv3_physics",
        "tendency_of_east_wind_due_to_fv3_physics",
        "tendency_of_north_wind_due_to_fv3_physics",
    ],
    times=DEFAULT_TIMES,
)
baseline_diagnostics = DiagnosticFile(
    name="diags.zarr",
    variables=["water_vapor_path", "physics_precip"],
    times=DEFAULT_TIMES,
)
