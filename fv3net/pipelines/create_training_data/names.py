# suffixes that denote whether diagnostic variable is from the coarsened
# high resolution prognostic run or the coarse res one step train data run
suffix_hires = "prog"
suffix_coarse_train = "train"

# variable names for one step run output and coarsened high res output
init_time_dim = "initial_time"
forecast_time_dim = "forecast_time"
step_time_dim = "step"
coord_begin_step = "begin"
var_lon_center, var_lat_center, var_lon_outer, var_lat_outer = (
    "lon",
    "lat",
    "lonb",
    "latb",
)
coord_x_center, coord_y_center, coord_z_center = ("x", "y", "z")
var_x_wind, var_y_wind = ("x_wind", "y_wind")
var_temp, var_sphum = ("air_temperature", "specific_humidity")
radiation_vars = [
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc",
]

one_step_vars = radiation_vars + [
    "total_precipitation",
    "surface_temperature",
    "land_sea_mask",
    "latent_heat_flux",
    "sensible_heat_flux",
    "mean_cos_zenith_angle",
    "surface_geopotential",
    "vertical_thickness_of_atmospheric_layer",
    "vertical_wind",
    "pressure_thickness_of_atmospheric_layer",
    var_temp,
    var_sphum,
    var_x_wind,
    var_y_wind,
]

# names for residuals that the ML is training on
# high resolution tendency - coarse res model's one step tendency
var_source_name_map = {
    var_x_wind: "dQU",
    var_y_wind: "dQV",
    var_temp: "dQ1",
    var_sphum: "dQ2",
}
target_vars = list(var_source_name_map.values())

# mappings for renaming of variables in training data output
renamed_high_res_vars = {
    **{f"{var}_coarse": f"{var}_{suffix_hires}" for var in radiation_vars},
    "lhtflsfc_coarse": f"latent_heat_flux_{suffix_hires}",
    "shtflsfc_coarse": f"sensible_heat_flux_{suffix_hires}",
}
renamed_one_step_vars = {var: f"{var}_{suffix_coarse_train}" for var in radiation_vars}
renamed_dims = {
    "grid_xt": "x",
    "grid_yt": "y",
    "grid_x": "x_interface",
    "grid_y": "y_interface",
}
