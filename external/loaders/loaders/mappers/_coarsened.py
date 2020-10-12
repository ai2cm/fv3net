from ._base import GeoMapper
import xarray as xr
import collections
import vcm
import os

RESTART_RENAMES = {
    "cx": "accumulated_x_courant_number",
    "mfx": "accumulated_x_mass_flux",
    "cy": "accumulated_y_courant_number",
    "mfy": "accumulated_y_mass_flux",
    "T": "air_temperature",
    "gt0": "air_temperature_after_physics",
    "t2m": "air_temperature_at_2m",
    "area": "area_of_grid_cell",
    "ak": "atmosphere_hybrid_a_coordinate",
    "bk": "atmosphere_hybrid_b_coordinate",
    "canopy": "canopy_water",
    "sfcflw": "total_sky_upward_longwave_flux_at_surface",
    "sfcfsw": "total_sky_upward_shortwave_flux_at_surface",
    "topflw": "total_sky_upward_longwave_flux_at_top_of_atmosphere",
    "topfsw": "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "cvb": "convective_cloud_bottom_pressure",
    "cv": "convective_cloud_fraction",
    "cvt": "convective_cloud_top_pressure",
    "tg3": "deep_soil_temperature",
    "diss_est": "dissipation_estimate_from_heat_source",
    "ua": "eastward_wind",
    "gu0": "eastward_wind_after_physics",
    "u_srf": "eastward_wind_at_surface",
    "ffhh": "fh_parameter",
    "f10m": "fm_at_10m",
    "ffmm": "fm_parameter",
    "facsf": "fractional_coverage_with_strong_cosz_dependency",
    "facwf": "fractional_coverage_with_weak_cosz_dependency",
    "uustar": "friction_velocity",
    "fice": "ice_fraction_over_open_water",
    "pe": "interface_pressure",
    "pk": "interface_pressure_raised_to_power_of_kappa",
    "slmsk": "land_sea_mask",
    "dqsfci": "latent_heat_flux",
    "xlat": "latitude",
    "pkz": "layer_mean_pressure_raised_to_power_of_kappa",
    "slc": "liquid_soil_moisture",
    "peln": "logarithm_of_interface_pressure",
    "xlon": "longitude",
    "shdmax": "maximum_fractional_coverage_of_green_vegetation",
    "snoalb": "maximum_snow_albedo_in_fraction",
    "coszen": "mean_cos_zenith_angle",
    "alnsf": "mean_near_infrared_albedo_with_strong_cosz_dependency",
    "alnwf": "mean_near_infrared_albedo_with_weak_cosz_dependency",
    "alvsf": "mean_visible_albedo_with_strong_cosz_dependency",
    "alvwf": "mean_visible_albedo_with_weak_cosz_dependency",
    "shdmin": "minimum_fractional_coverage_of_green_vegetation",
    "va": "northward_wind",
    "gv0": "northward_wind_after_physics",
    "v_srf": "northward_wind_at_surface",
    "delp": "pressure_thickness_of_atmospheric_layer",
    "hice": "sea_ice_thickness",
    "dtsfci": "sensible_heat_flux",
    "sncovr": "snow_cover_in_fraction",
    "snwdph": "snow_depth_water_equivalent",
    "srflag": "snow_rain_flag",
    "stc": "soil_temperature",
    "stype": "soil_type",
    "q2m": "specific_humidity_at_2m",
    "phis": "surface_geopotential",
    "ps": "surface_pressure",
    "zorl": "surface_roughness",
    "slope": "surface_slope_type",
    "tsea": "surface_temperature",
    "tisfc": "surface_temperature_over_ice_fraction",
    "q_con": "total_condensate_mixing_ratio",
    "tprcp": "total_precipitation",
    "smc": "total_soil_moisture",
    "vfrac": "vegetation_fraction",
    "vtype": "vegetation_type",
    "omga": "vertical_pressure_velocity",
    "DZ": "vertical_thickness_of_atmospheric_layer",
    "W": "vertical_wind",
    "sheleg": "water_equivalent_of_accumulated_snow_depth",
    "u": "x_wind",
    "uc": "x_wind_on_c_grid",
    "v": "y_wind",
    "vc": "y_wind_on_c_grid",
    "grid_xt": "x",
    "grid_yt": "y",
    "pfull": "z",
    "sphum": "specific_humidity",
}


class CoarsenedDataMapper(GeoMapper, collections.abc.Mapping):
    def __init__(self, url):
        self._url = url
        self._fs = vcm.cloud.get_fs(url)
        self._keys = None
        self._n = 0

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = vcm.open_restarts(os.path.join(self._url, key))
        ds = ds.squeeze("file_prefix")
        renames = {
            name: value for (name, value) in RESTART_RENAMES.items() if name in ds
        }
        ds = ds.rename(renames)
        return ds

    def keys(self):
        if self._keys is None:
            self._keys = sorted(
                list(os.path.basename(fname) for fname in self._fs.ls(self._url))
            )
        return self._keys

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n <= len(self):
            result = list(self.keys)[self._n]
            self._n += 1
            return result
        else:
            raise StopIteration
