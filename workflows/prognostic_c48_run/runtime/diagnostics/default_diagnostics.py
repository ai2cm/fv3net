from .manager import DiagnosticFileConfig
from .fortran import FortranFileConfig, FortranVariableNameSpec

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
sfc_dt_atmos = FortranFileConfig(
    name="sfc_dt_atmos.zarr",
    chunks={"time": 96},
    variables=[
        FortranVariableNameSpec("dynamics", "grid_lont", "lon"),
        FortranVariableNameSpec("dynamics", "grid_latt", "lat"),
        FortranVariableNameSpec("dynamics", "grid_lon", "lonb"),
        FortranVariableNameSpec("dynamics", "grid_lat", "latb"),
        FortranVariableNameSpec("dynamics", "area", "area"),
        FortranVariableNameSpec("gfs_phys", "dusfci", "uflx"),
        FortranVariableNameSpec("gfs_phys", "dvsfci", "vflx"),
        FortranVariableNameSpec("gfs_phys", "cnvprcpb_ave", "CPRATsfc"),
        FortranVariableNameSpec("gfs_phys", "totprcpb_ave", "PRATEsfc"),
        FortranVariableNameSpec("gfs_phys", "toticeb_ave", "ICEsfc"),
        FortranVariableNameSpec("gfs_phys", "totsnwb_ave", "SNOWsfc"),
        FortranVariableNameSpec("gfs_phys", "totgrpb_ave", "GRAUPELsfc"),
        FortranVariableNameSpec("gfs_phys", "DSWRF", "DSWRFsfc"),
        FortranVariableNameSpec("gfs_phys", "USWRF", "USWRFsfc"),
        FortranVariableNameSpec("gfs_phys", "DSWRFtoa", "DSWRFtoa"),
        FortranVariableNameSpec("gfs_phys", "USWRFtoa", "USWRFtoa"),
        FortranVariableNameSpec("gfs_phys", "ULWRFtoa", "ULWRFtoa"),
        FortranVariableNameSpec("gfs_phys", "ULWRF", "ULWRFsfc"),
        FortranVariableNameSpec("gfs_phys", "DLWRF", "DLWRFsfc"),
        FortranVariableNameSpec("gfs_phys", "lhtfl_ave", "LHTFLsfc"),
        FortranVariableNameSpec("gfs_phys", "shtfl_ave", "SHTFLsfc"),
        FortranVariableNameSpec("gfs_phys", "hpbl", "HPBLsfc"),
        FortranVariableNameSpec("gfs_sfc", "fice", "ICECsfc"),
        FortranVariableNameSpec("gfs_sfc", "SLMSKsfc", "SLMSKsfc"),
        FortranVariableNameSpec("gfs_sfc", "q2m", "SPFH2m"),
        FortranVariableNameSpec("gfs_sfc", "t2m", "TMP2m"),
        FortranVariableNameSpec("gfs_sfc", "tsfc", "TMPsfc"),
        FortranVariableNameSpec("gfs_phys", "dpt2m", "DPT2m"),
        FortranVariableNameSpec("gfs_phys", "u10m", "UGRD10m"),
        FortranVariableNameSpec("gfs_phys", "v10m", "VGRD10m"),
        FortranVariableNameSpec("gfs_phys", "tmpmax2m", "TMAX2m"),
        FortranVariableNameSpec("gfs_phys", "wind10mmax", "MAXWIND10m"),
        FortranVariableNameSpec("gfs_phys", "soilm", "SOILM"),
        FortranVariableNameSpec("gfs_sfc", "SOILT1", "SOILT1"),
        FortranVariableNameSpec("gfs_sfc", "SOILT2", "SOILT2"),
        FortranVariableNameSpec("gfs_sfc", "SOILT3", "SOILT3"),
        FortranVariableNameSpec("gfs_sfc", "SOILT4", "SOILT4"),
    ],
)
atmos_dt_atmos = FortranFileConfig(
    name="atmos_dt_atmos.zarr",
    chunks={"time": 96},
    variables=[
        FortranVariableNameSpec("dynamics", "grid_lont", "lon"),
        FortranVariableNameSpec("dynamics", "grid_latt", "lat"),
        FortranVariableNameSpec("dynamics", "grid_lon", "lonb"),
        FortranVariableNameSpec("dynamics", "grid_lat", "latb"),
        FortranVariableNameSpec("dynamics", "area", "area"),
        FortranVariableNameSpec("dynamics", "us", "UGRDlowest"),
        FortranVariableNameSpec("dynamics", "u850", "UGRD850"),
        FortranVariableNameSpec("dynamics", "u500", "UGRD500"),
        FortranVariableNameSpec("dynamics", "u200", "UGRD200"),
        FortranVariableNameSpec("dynamics", "u50", "UGRD50"),
        FortranVariableNameSpec("dynamics", "vs", "VGRDlowest"),
        FortranVariableNameSpec("dynamics", "v850", "VGRD850"),
        FortranVariableNameSpec("dynamics", "v500", "VGRD500"),
        FortranVariableNameSpec("dynamics", "v200", "VGRD200"),
        FortranVariableNameSpec("dynamics", "v50", "VGRD50"),
        FortranVariableNameSpec("dynamics", "tm", "TMP500_300"),
        FortranVariableNameSpec("dynamics", "tb", "TMPlowest"),
        FortranVariableNameSpec("dynamics", "t850", "TMP850"),
        FortranVariableNameSpec("dynamics", "t500", "TMP500"),
        FortranVariableNameSpec("dynamics", "t200", "TMP200"),
        FortranVariableNameSpec("dynamics", "w850", "w850"),
        FortranVariableNameSpec("dynamics", "w500", "w500"),
        FortranVariableNameSpec("dynamics", "w200", "w200"),
        FortranVariableNameSpec("dynamics", "w50", "w50"),
        FortranVariableNameSpec("dynamics", "vort850", "VORT850"),
        FortranVariableNameSpec("dynamics", "vort500", "VORT500"),
        FortranVariableNameSpec("dynamics", "vort200", "VORT200"),
        FortranVariableNameSpec("dynamics", "z850", "h850"),
        FortranVariableNameSpec("dynamics", "z500", "h500"),
        FortranVariableNameSpec("dynamics", "z200", "h200"),
        FortranVariableNameSpec("dynamics", "rh1000", "RH1000"),
        FortranVariableNameSpec("dynamics", "rh925", "RH925"),
        FortranVariableNameSpec("dynamics", "rh850", "RH850"),
        FortranVariableNameSpec("dynamics", "rh700", "RH700"),
        FortranVariableNameSpec("dynamics", "rh500", "RH500"),
        FortranVariableNameSpec("dynamics", "q1000", "q1000"),
        FortranVariableNameSpec("dynamics", "q925", "q925"),
        FortranVariableNameSpec("dynamics", "q850", "q850"),
        FortranVariableNameSpec("dynamics", "q700", "q700"),
        FortranVariableNameSpec("dynamics", "q500", "q500"),
        FortranVariableNameSpec("dynamics", "slp", "PRMSL"),
        FortranVariableNameSpec("dynamics", "ps", "PRESsfc"),
        FortranVariableNameSpec("dynamics", "tq", "PWAT"),
        FortranVariableNameSpec("dynamics", "lw", "VIL"),
        FortranVariableNameSpec("dynamics", "iw", "iw"),
        FortranVariableNameSpec("dynamics", "ke", "kinetic_energy"),
        FortranVariableNameSpec("dynamics", "te", "total_energy"),
    ],
)
atmos_8xdaily = FortranFileConfig(
    name="atmos_8xdaily.zarr",
    chunks={"time": 8},
    variables=[
        FortranVariableNameSpec("dynamics", "grid_lont", "lon"),
        FortranVariableNameSpec("dynamics", "grid_latt", "lat"),
        FortranVariableNameSpec("dynamics", "grid_lon", "lonb"),
        FortranVariableNameSpec("dynamics", "grid_lat", "latb"),
        FortranVariableNameSpec("dynamics", "area", "area"),
        FortranVariableNameSpec("dynamics", "ucomp", "ucomp"),
        FortranVariableNameSpec("dynamics", "vcomp", "vcomp"),
        FortranVariableNameSpec("dynamics", "temp", "temp"),
        FortranVariableNameSpec("dynamics", "delp", "delp"),
        FortranVariableNameSpec("dynamics", "sphum", "sphum"),
        FortranVariableNameSpec("dynamics", "pfnh", "nhpres"),
        FortranVariableNameSpec("dynamics", "w", "w"),
        FortranVariableNameSpec("dynamics", "delz", "delz"),
        FortranVariableNameSpec("dynamics", "ps", "ps"),
        FortranVariableNameSpec("dynamics", "reflectivity", "reflectivity"),
        FortranVariableNameSpec("dynamics", "liq_wat", "liq_wat"),
        FortranVariableNameSpec("dynamics", "ice_wat", "ice_wat"),
        FortranVariableNameSpec("dynamics", "rainwat", "rainwat"),
        FortranVariableNameSpec("dynamics", "snowwat", "snowwat"),
        FortranVariableNameSpec("dynamics", "graupel", "graupel"),
    ],
)
nudging_tendencies_fortran = FortranFileConfig(
    name="nudging_tendencies.zarr",
    chunks={"time": 8},
    variables=[
        FortranVariableNameSpec("dynamics", "grid_lont", "lon"),
        FortranVariableNameSpec("dynamics", "grid_latt", "lat"),
        FortranVariableNameSpec("dynamics", "grid_lon", "lonb"),
        FortranVariableNameSpec("dynamics", "grid_lat", "latb"),
        FortranVariableNameSpec("dynamics", "area", "area"),
        FortranVariableNameSpec("dynamics", "u_dt_nudge", "u_dt_nudge"),
        FortranVariableNameSpec("dynamics", "v_dt_nudge", "v_dt_nudge"),
        FortranVariableNameSpec("dynamics", "delp_dt_nudge", "delp_dt_nudge"),
        FortranVariableNameSpec("dynamics", "ps_dt_nudge", "ps_dt_nudge"),
        FortranVariableNameSpec("dynamics", "t_dt_nudge", "t_dt_nudge"),
        FortranVariableNameSpec("dynamics", "q_dt_nudge", "q_dt_nudge"),
    ],
)
