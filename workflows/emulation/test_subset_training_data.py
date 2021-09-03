import sys
import runpy


def subset_config():
    return """source_path: gs://vcm-ml-experiments/andrep/2021-05-28/training-simple-phys-12day-w-micro/ # noqa
destination_path: gs://vcm-ml-experiments/andrep/2021-05-28/training-subsets/simple-phys-hybridedmf-w-microphysics-12day # noqa
source_files:
  - state_after_dynamics.zarr
  - physics_tendencies.zarr
  - physics_component_tendencies.zarr
init_times: 
  - "20160101.000000"
  - "20160301.000000"
  - "20160401.000000"
  - "20160501.000000"
  - "20160701.000000"
  - "20160801.000000"
  - "20161001.000000"
  - "20161101.000000"
  - "20161201.000000"
variables: 
  - cos_day
  - sin_day
  - cos_month
  - sin_month
  - latitude
  - longitude
  - cos_lon
  - sin_lon
  - cos_zenith_angle
  - vertical_thickness_of_atmospheric_layer
  - pressure_thickness_of_atmospheric_layer
  - eastward_wind
  - northward_wind
  - vertical_wind
  - air_temperature
  - specific_humidity
  - cloud_water_mixing_ratio
  - ozone_mixing_ratio
  - total_soil_moisture
  - liquid_soil_moisture
  - soil_temperature
  - surface_temperature
  - canopy_water
  - sea_ice_thickness
  - snow_depth_water_equivalent
  - tendency_of_air_temperature_due_to_fv3_physics
  - tendency_of_specific_humidity_due_to_fv3_physics
  - tendency_of_eastward_wind_due_to_fv3_physics
  - tendency_of_northward_wind_due_to_fv3_physics
  - tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics
  - tendency_of_ozone_mixing_ratio_due_to_fv3_physics
  - tendency_of_pressure_thickness_of_atmospheric_layer_due_to_fv3_physics
  - tendency_of_specific_humidity_due_to_microphysics
  - tendency_of_air_temperature_due_to_microphysics
"""


def test_subset_training_data(tmpdir, monkeypatch):

    config = tmpdir.join("config")
    with config.open("w") as f:
        f.write(subset_config())

    monkeypatch.setattr(sys, "argv", ["", str(config)])
    runpy.run_path("subset_training_data.py", run_name="__main__")
