# TODO this utility should be refactored to shared micropackage or vcm
# right now kubernetes and fv3config are added to setup.py
# for this dependency- when refactored, remove them from there
from fv3net.pipelines.common import update_nested_dict

# TODO revamp this default dictionary to be more declaretive. It should specify:
# what the output variables names are, and where each comes from e.g.:
#
# - name: surface_geopotential
#   rename:
#     from: one_step
#     name: surface_geopotential # maybe infer from above
# - name: PRATEsfc_highres
#   rename:
#     from: high_res
#     name: PRATEsfc_coarse
# - name: dQU
#   apparent_source:
#     variable: x_wind
#
# I suspect this could be pretty easily parsed into the format below,
# or the code restructed to use this instead.


DEFAULT = {
    "suffix_hires": "prog",
    "suffix_coarse_train": "train",
    "var_source_name_map": {
        "x_wind": "dQU",
        "y_wind": "dQV",
        "air_temperature": "dQ1",
        "specific_humidity": "dQ2",
    },
    "init_time_dim": "initial_time",
    "forecast_time_dim": "forecast_time",
    "step_time_dim": "step",
    "step_for_state": "after_physics",
    "coord_begin_step": "begin",
    "coord_before_physics": "after_dynamics",
    "coord_after_physics": "after_physics",
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_z_center": "z",
    "var_lon_center": "lon",
    "var_lat_center": "lat",
    "var_lon_outer": "lonb",
    "var_lat_outer": "latb",
    "physics_tendency_names": {"specific_humidity": "pQ2", "air_temperature": "pQ1"},
    "renamed_dims": {
        "grid_xt": "x",
        "grid_yt": "y",
        "grid_x": "x_interface",
        "grid_y": "y_interface",
        "initialization_time": "initial_time",
    },
    "edge_dims": ["x_interface", "y_interface"],
    "edge_to_center_dims": {"x_interface": "x", "y_interface": "y"},
    "grid_vars": ["area", "latb", "lonb", "lat", "lon"],
    "var_x_wind": "x_wind",
    "var_y_wind": "y_wind",
    "var_temp": "air_temperature",
    "var_sphum": "specific_humidity",
    "var_land_sea_mask": "land_sea_mask",
    "one_step_vars": [
        "latent_heat_flux",
        "sensible_heat_flux",
        "total_precipitation",
        "dQU",
        "dQV",
        "dQ1",
        "dQ2",
        "pQ1",
        "pQ2",
        "surface_temperature",
        "land_sea_mask",
        "surface_geopotential",
        "vertical_thickness_of_atmospheric_layer",
        "vertical_wind",
        "pressure_thickness_of_atmospheric_layer",
        "specific_humidity",
        "air_temperature",
        "DSWRFtoa_train",
        "DSWRFsfc_train",
        "USWRFtoa_train",
        "USWRFsfc_train",
        "DLWRFsfc_train",
        "ULWRFtoa_train",
        "ULWRFsfc_train",
        "lat",
        "lon",
        "latb",
        "lonb",
        "area",
    ],
    "renamed_high_res_data_variables": {
        "DSWRFtoa_coarse": "DSWRFtoa_prog",
        "DSWRFsfc_coarse": "DSWRFsfc_prog",
        "USWRFtoa_coarse": "USWRFtoa_prog",
        "USWRFsfc_coarse": "USWRFsfc_prog",
        "DLWRFsfc_coarse": "DLWRFsfc_prog",
        "ULWRFtoa_coarse": "ULWRFtoa_prog",
        "ULWRFsfc_coarse": "ULWRFsfc_prog",
        "SHTFLsfc_coarse": "sensible_heat_flux_prog",
        "LHTFLsfc_coarse": "latent_heat_flux_prog",
        "PRATEsfc_coarse": "total_precipitation_prog",
},
    "diag_vars": [
        "DSWRFtoa",
        "DSWRFsfc",
        "USWRFtoa",
        "USWRFsfc",
        "DLWRFsfc",
        "ULWRFtoa",
        "ULWRFsfc",
    ],
}


def get_config(updates):
    return update_nested_dict(DEFAULT, updates)
