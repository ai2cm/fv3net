physics_url = "gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr"  # noqa
restart_url = "gs://vcm-ml-experiments/2020-06-02-fine-res/2020-05-27-40-day-X-SHiELD-simulation-C384-restart-files.zarr"  # noqa
gfsphysics_url = "gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/gfsphysics_15min_coarse.zarr/"  # noqa
area_url = "gs://vcm-ml-raw/2020-11-10-C3072-to-C384-exposed-area.zarr"

# Input data specifications
PHYSICS_VARIABLES = [
    "t_dt_fv_sat_adj_coarse",
    "t_dt_nudge_coarse",
    "t_dt_phys_coarse",
    "qv_dt_fv_sat_adj_coarse",
    "qv_dt_phys_coarse",
    "eddy_flux_vulcan_omega_sphum",
    "eddy_flux_vulcan_omega_temp",
    "vulcan_omega_coarse",
    "area_coarse",
]


RESTART_VARIABLES = [
    "delp",
    "sphum",
    "T",
]

GFSPHYSICS_VARIABLES = [
    "dq3dt_deep_conv_coarse",
    "dq3dt_mp_coarse",
    "dq3dt_pbl_coarse",
    "dq3dt_shal_conv_coarse",
    "dt3dt_deep_conv_coarse",
    "dt3dt_lw_coarse",
    "dt3dt_mp_coarse",
    "dt3dt_ogwd_coarse",
    "dt3dt_pbl_coarse",
    "dt3dt_shal_conv_coarse",
    "dt3dt_sw_coarse",
]

# Output configurations
VARIABLES_TO_AVERAGE = set(
    GFSPHYSICS_VARIABLES + PHYSICS_VARIABLES + RESTART_VARIABLES
) - {"area_coarse", "delp"}
# coarsening factor. C384 to C48 is a factor of 8
factor = 8
