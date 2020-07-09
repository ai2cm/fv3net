SAMPLE_DIM_NAME = "sample"
TIME_NAME = "time"
TIME_FMT = "%Y%m%d.%H%M%S"
DERIVATION_DIM = "derivation"
DERIVATION_SHIELD_COORD = "coarsened_SHiELD"
DERIVATION_FV3GFS_COORD = "coarse_FV3GFS"
RENAMED_SHIELD_DIAG_VARS = {
    "DSWRFtoa_coarse": "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    "DSWRFsfc_coarse": "total_sky_downward_shortwave_flux_at_surface",
    "USWRFtoa_coarse": "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "USWRFsfc_coarse": "total_sky_upward_shortwave_flux_at_surface",
    "DLWRFsfc_coarse": "total_sky_downward_longwave_flux_at_surface",
    "ULWRFtoa_coarse": "total_sky_upward_longwave_flux_at_top_of_atmosphere",
    "ULWRFsfc_coarse": "total_sky_upward_longwave_flux_at_surface",
    "SHTFLsfc_coarse": "sensible_heat_flux",
    "LHTFLsfc_coarse": "latent_heat_flux",
    "PRATEsfc_coarse": "surface_precipitation_rate",
}
