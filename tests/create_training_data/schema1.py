
from zarr_to_test_schema import DatasetSchema, CoordinateSchema, VariableSchema, Range
import numpy as np


def create_schema() -> DatasetSchema:

    sfc_domain = Range(np.float64, min=0, max=1000)
    return DatasetSchema(
        coords=[
            CoordinateSchema(
                name="forecast_time",
                dims=["forecast_time"],
                value=np.array(
                    [
                        0.0,
                        60.0,
                        120.0,
                        180.0,
                        240.0,
                        300.0,
                        360.0,
                        420.0,
                        480.0,
                        540.0,
                        600.0,
                        660.0,
                        720.0,
                        780.0,
                        840.0,
                    ]
                ),
            ),
            CoordinateSchema(
                name="initial_time",
                dims=["initial_time"],
                value=np.array(
                    [
                        "20160801.001500",
                        "20160801.003000",
                        "20160801.004500",
                        "20160801.010000",
                        "20160801.011500",
                        "20160801.013000",
                    ],
                    dtype="<U15",
                ),
            ),
            CoordinateSchema(
                name="step",
                dims=["step"],
                value=np.array(["begin", "after_physics", "after_dynamics"], dtype="<U14"),
            ),
        ],
        variables=[
            VariableSchema(
                name="DLWRFsfc",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=sfc_domain
            ),
            VariableSchema(
                name="DSWRFsfc",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=sfc_domain
            ),
            VariableSchema(
                name="DSWRFtoa",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=sfc_domain
            ),
            VariableSchema(
                name="ULWRFsfc",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=sfc_domain
            ),
            VariableSchema(
                name="ULWRFtoa",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=sfc_domain
            ),
            VariableSchema(
                name="USWRFsfc",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=sfc_domain
            ),
            VariableSchema(
                name="USWRFtoa",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=sfc_domain
            ),
            VariableSchema(
                name="air_temperature",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=188.81748962402344, max=313.4430847167969
                ),
            ),
            VariableSchema(
                name="air_temperature_at_2m",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=274.8279113769531, max=311.18475341796875
                ),
            ),
            VariableSchema(
                name="area",
                dims=["initial_time", "tile", "y", "x"],
                shape=(6, 6, 48, 48),
                domain=Range(dtype=np.dtype("float32"), min=23545550000.0, max=23545550000.0),
            ),
            VariableSchema(
                name="canopy_water",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.4610218405723572),
            ),
            VariableSchema(
                name="cloud_amount",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=-0.03345150128006935, max=1.0),
            ),
            VariableSchema(
                name="cloud_ice_mixing_ratio",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=-5.430288396723881e-08,
                    max=0.0008673568372614682,
                ),
            ),
            VariableSchema(
                name="cloud_water_mixing_ratio",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=-6.105483771534637e-05,
                    max=0.0007919442723505199,
                ),
            ),
            VariableSchema(
                name="convective_cloud_bottom_pressure",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="convective_cloud_fraction",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="convective_cloud_top_pressure",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="deep_soil_temperature",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=282.4851989746094, max=302.2434387207031
                ),
            ),
            VariableSchema(
                name="eastward_wind_at_surface",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=-10.438079833984375, max=13.63209342956543
                ),
            ),
            VariableSchema(
                name="fh_parameter",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=2.545614004135132, max=29.177534103393555
                ),
            ),
            VariableSchema(
                name="fm_at_10m",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.7755287289619446, max=0.9923458695411682
                ),
            ),
            VariableSchema(
                name="fm_parameter",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=2.4025447368621826, max=28.077821731567383
                ),
            ),
            VariableSchema(
                name="fractional_coverage_with_strong_cosz_dependency",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=1.0000001192092896),
            ),
            VariableSchema(
                name="fractional_coverage_with_weak_cosz_dependency",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=-7.756063149506527e-18,
                    max=1.0000001192092896,
                ),
            ),
            VariableSchema(
                name="friction_velocity",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.02910376340150833, max=0.8100247979164124
                ),
            ),
            VariableSchema(
                name="graupel_mixing_ratio",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=-1.8559602210643789e-07,
                    max=0.0004039166960865259,
                ),
            ),
            VariableSchema(
                name="ice_fraction_over_open_water",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="land_sea_mask",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=1.0),
            ),
            VariableSchema(
                name="lat",
                dims=["initial_time", "tile", "y", "x"],
                shape=(6, 6, 48, 48),
                domain=Range(dtype=np.dtype("float32"), min=-34.891106, max=-34.891106),
            ),
            VariableSchema(
                name="latb",
                dims=["initial_time", "tile", "y_interface", "x_interface"],
                shape=(6, 6, 49, 49),
                domain=Range(dtype=np.dtype("float32"), min=-35.26439, max=-35.26439),
            ),
            VariableSchema(
                name="latent_heat_flux",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="liquid_soil_moisture",
                dims=["initial_time", "step", "forecast_time", "tile", "z_soil", "y", "x"],
                shape=(6, 3, 15, 6, 4, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.02517462708055973, max=1.0000001192092896
                ),
            ),
            VariableSchema(
                name="lon",
                dims=["initial_time", "tile", "y", "x"],
                shape=(6, 6, 48, 48),
                domain=Range(dtype=np.dtype("float32"), min=305.7829, max=305.7829),
            ),
            VariableSchema(
                name="lonb",
                dims=["initial_time", "tile", "y_interface", "x_interface"],
                shape=(6, 6, 49, 49),
                domain=Range(dtype=np.dtype("float32"), min=305.0, max=305.0),
            ),
            VariableSchema(
                name="maximum_fractional_coverage_of_green_vegetation",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.9900000095367432),
            ),
            VariableSchema(
                name="maximum_snow_albedo_in_fraction",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.706956684589386),
            ),
            VariableSchema(
                name="mean_cos_zenith_angle",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="mean_near_infrared_albedo_with_strong_cosz_dependency",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.059999991208314896, max=0.6255454421043396
                ),
            ),
            VariableSchema(
                name="mean_near_infrared_albedo_with_weak_cosz_dependency",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.059999991208314896, max=0.6240035891532898
                ),
            ),
            VariableSchema(
                name="mean_visible_albedo_with_strong_cosz_dependency",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.02158251777291298, max=0.3649003207683563
                ),
            ),
            VariableSchema(
                name="mean_visible_albedo_with_weak_cosz_dependency",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.020436571910977364, max=0.3603686988353729
                ),
            ),
            VariableSchema(
                name="minimum_fractional_coverage_of_green_vegetation",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.699999988079071),
            ),
            VariableSchema(
                name="ozone_mixing_ratio",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=3.944625603935492e-08,
                    max=1.715179496386554e-05,
                ),
            ),
            VariableSchema(
                name="pressure_thickness_of_atmospheric_layer",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=281.4435729980469, max=1944.5164794921875
                ),
            ),
            VariableSchema(
                name="rain_mixing_ratio",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=-2.613103093282443e-12,
                    max=0.00029887654818594456,
                ),
            ),
            VariableSchema(
                name="sea_ice_thickness",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="sensible_heat_flux",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.0),
            ),
            VariableSchema(
                name="snow_cover_in_fraction",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.1280592381954193),
            ),
            VariableSchema(
                name="snow_depth_water_equivalent",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=8.609997749328613),
            ),
            VariableSchema(
                name="snow_mixing_ratio",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=-1.714159703070095e-09,
                    max=0.0008146102773025632,
                ),
            ),
            VariableSchema(
                name="snow_rain_flag",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=1.0),
            ),
            VariableSchema(
                name="soil_temperature",
                dims=["initial_time", "step", "forecast_time", "tile", "z_soil", "y", "x"],
                shape=(6, 3, 15, 6, 4, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=277.7989807128906, max=314.1700439453125
                ),
            ),
            VariableSchema(
                name="soil_type",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=12.0),
            ),
            VariableSchema(
                name="specific_humidity",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=3.2631128021876066e-08,
                    max=0.01869734190404415,
                ),
            ),
            VariableSchema(
                name="specific_humidity_at_2m",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=0.0008022307883948088,
                    max=0.019105618819594383,
                ),
            ),
            VariableSchema(
                name="surface_geopotential",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=-9.078812599182129, max=18575.51953125
                ),
            ),
            VariableSchema(
                name="surface_roughness",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=0.00012781869736500084,
                    max=265.29998779296875,
                ),
            ),
            VariableSchema(
                name="surface_slope_type",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=6.0),
            ),
            VariableSchema(
                name="surface_temperature",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=272.9617919921875, max=309.5419006347656
                ),
            ),
            VariableSchema(
                name="surface_temperature_over_ice_fraction",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=272.9617919921875, max=309.5419006347656
                ),
            ),
            VariableSchema(
                name="total_precipitation",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.00018706226546783),
            ),
            VariableSchema(
                name="total_soil_moisture",
                dims=["initial_time", "step", "forecast_time", "tile", "z_soil", "y", "x"],
                shape=(6, 3, 15, 6, 4, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=0.02517462708055973, max=1.0000001192092896
                ),
            ),
            VariableSchema(
                name="vegetation_fraction",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=0.9025070071220398),
            ),
            VariableSchema(
                name="vegetation_type",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=16.0),
            ),
            VariableSchema(
                name="vertical_thickness_of_atmospheric_layer",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=-5624.0244140625, max=-28.116539001464844
                ),
            ),
            VariableSchema(
                name="vertical_wind",
                dims=["initial_time", "step", "forecast_time", "tile", "z", "y", "x"],
                shape=(6, 3, 15, 6, 79, 48, 48),
                domain=Range(
                    dtype=np.dtype("float64"),
                    min=-0.17804476618766785,
                    max=0.40060845017433167,
                ),
            ),
            VariableSchema(
                name="water_equivalent_of_accumulated_snow_depth",
                dims=["initial_time", "step", "forecast_time", "tile", "y", "x"],
                shape=(6, 3, 15, 6, 48, 48),
                domain=Range(dtype=np.dtype("float64"), min=0.0, max=7.761899471282959),
            ),
            VariableSchema(
                name="x_wind",
                dims=[
                    "initial_time",
                    "step",
                    "forecast_time",
                    "tile",
                    "z",
                    "y_interface",
                    "x",
                ],
                shape=(6, 3, 15, 6, 79, 49, 48),
                domain=Range(
                    dtype=np.dtype("float64"), min=-44.940101623535156, max=113.11870574951172
                ),
            ),
            VariableSchema(
                name="y_wind",
                dims=[
                    "initial_time",
                    "step",
                    "forecast_time",
                    "tile",
                    "z",
                    "y",
                    "x_interface",
                ],
                shape=(6, 3, 15, 6, 79, 48, 49),
                domain=Range(
                    dtype=np.dtype("float64"), min=-40.80194091796875, max=46.32950210571289
                ),
            ),
        ],
    )
