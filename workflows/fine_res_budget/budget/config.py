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

ATMOS_AVG_VARIABLES = [
    "delp_dt_nudge_coarse",
    "ice_wat_dt_gfdlmp_coarse",
    "ice_wat_dt_phys_coarse",
    "liq_wat_dt_gfdlmp_coarse",
    "liq_wat_dt_phys_coarse",
    "qg_dt_gfdlmp_coarse",
    "qg_dt_phys_coarse",
    "qi_dt_gfdlmp_coarse",
    "qi_dt_phys_coarse",
    "ql_dt_gfdlmp_coarse",
    "ql_dt_phys_coarse",
    "qr_dt_gfdlmp_coarse",
    "qr_dt_phys_coarse",
    "qs_dt_gfdlmp_coarse",
    "qs_dt_phys_coarse",
    "qv_dt_gfdlmp_coarse",
    "qv_dt_phys_coarse",
    "t_dt_gfdlmp_coarse",
    "t_dt_nudge_coarse",
    "t_dt_phys_coarse",
    "u_dt_gfdlmp_coarse",
    "u_dt_nudge_coarse",
    "u_dt_phys_coarse",
    "v_dt_gfdlmp_coarse",
    "v_dt_nudge_coarse",
    "v_dt_phys_coarse",
]

RESTART_VARIABLES = [
    "delp",
    "sphum",
    "T",
]


# Output configurations
VARIABLES_TO_AVERAGE = set(
    ATMOS_AVG_VARIABLES + PHYSICS_VARIABLES + RESTART_VARIABLES
) - {"area_coarse", "delp"}
# coarsening factor. C384 to C48 is a factor of 8
factor = 8
