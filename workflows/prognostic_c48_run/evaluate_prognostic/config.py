from pathlib import Path
from runtime.segmented_run.prepare_config import to_fv3config
import fv3config
import fv3kube


fv3config_path = Path(__file__).parent / "fv3config.yml"


def get_config():
    with open(fv3config_path) as f:
        config = fv3config.load(f)

    restart_url = "gs://vcm-ml-experiments/online-emulator/2021-08-09/gfs-initialized-baseline-08/fv3gfs_run/artifacts/20160801.000000/RESTART"  # noqa

    dict_ = fv3kube.merge_fv3config_overlays(
        fv3config.enable_restart(config, restart_url), _get_overlay()
    )

    del dict_["nudging"]

    return to_fv3config(dict_, nudging_url="")


def _get_fortran_diagnostics():
    return [
        {
            "name": "sfc_dt_atmos.zarr",
            "chunks": {"time": 12},
            "times": {"frequency": 900, "kind": "interval"},
            "variables": [
                {
                    "module_name": "dynamics",
                    "field_name": "grid_lont",
                    "output_name": "lon",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "grid_latt",
                    "output_name": "lat",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "grid_lon",
                    "output_name": "lonb",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "grid_lat",
                    "output_name": "latb",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "area",
                    "output_name": "area",
                },
                {
                    "module_name": "gfs_phys",
                    "field_name": "cnvprcpb_ave",
                    "output_name": "convective_precipitation_diagnostic",
                },
                {
                    "module_name": "gfs_phys",
                    "field_name": "totprcpb_ave",
                    "output_name": "total_precipitation_diagnostic",
                },
                {
                    "module_name": "gfs_phys",
                    "field_name": "lhtfl_ave",
                    "output_name": "latent_heat_flux_diagnostic",
                },
                {
                    "module_name": "gfs_phys",
                    "field_name": "shtfl_ave",
                    "output_name": "sensible_heat_flux_diagnostics",
                },
                {
                    "module_name": "gfs_phys",
                    "field_name": "soilm",
                    "output_name": "SOILM",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "us",
                    "output_name": "UGRDlowest",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "u850",
                    "output_name": "UGRD850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "u500",
                    "output_name": "UGRD500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "u200",
                    "output_name": "UGRD200",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "u50",
                    "output_name": "UGRD50",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "vs",
                    "output_name": "VGRDlowest",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "v850",
                    "output_name": "VGRD850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "v500",
                    "output_name": "VGRD500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "v200",
                    "output_name": "VGRD200",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "v50",
                    "output_name": "VGRD50",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "tm",
                    "output_name": "TMP500_300",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "tb",
                    "output_name": "TMPlowest",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "t850",
                    "output_name": "TMP850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "t500",
                    "output_name": "TMP500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "t200",
                    "output_name": "TMP200",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "w850",
                    "output_name": "w850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "w500",
                    "output_name": "w500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "w200",
                    "output_name": "w200",
                },
                {"module_name": "dynamics", "field_name": "w50", "output_name": "w50"},
                {
                    "module_name": "dynamics",
                    "field_name": "vort850",
                    "output_name": "VORT850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "vort500",
                    "output_name": "VORT500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "vort200",
                    "output_name": "VORT200",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "z850",
                    "output_name": "h850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "z500",
                    "output_name": "h500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "z200",
                    "output_name": "h200",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "rh1000",
                    "output_name": "RH1000",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "rh925",
                    "output_name": "RH925",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "rh850",
                    "output_name": "RH850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "rh700",
                    "output_name": "RH700",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "rh500",
                    "output_name": "RH500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "q1000",
                    "output_name": "q1000",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "q925",
                    "output_name": "q925",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "q850",
                    "output_name": "q850",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "q700",
                    "output_name": "q700",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "q500",
                    "output_name": "q500",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "slp",
                    "output_name": "PRMSL",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "ps",
                    "output_name": "PRESsfc",
                },
                {"module_name": "dynamics", "field_name": "tq", "output_name": "PWAT"},
                {"module_name": "dynamics", "field_name": "lw", "output_name": "VIL"},
                {"module_name": "dynamics", "field_name": "iw", "output_name": "iw"},
                {
                    "module_name": "dynamics",
                    "field_name": "ke",
                    "output_name": "kinetic_energy",
                },
                {
                    "module_name": "dynamics",
                    "field_name": "te",
                    "output_name": "total_energy",
                },
            ],
        }
    ]


def _get_overlay():
    return {
        "base_version": "v0.5",
        "online_emulator": {
            "emulator": "dummy/path",
            "train": False,
            "online": True,
            "ignore_humidity_below": None,
        },
        "fortran_diagnostics": _get_fortran_diagnostics(),
        "namelist": {
            "coupler_nml": {"days": 0, "hours": 3, "minutes": 0, "seconds": 0},
            "diag_manager_nml": {"flush_nc_files": True},
        },
        "diagnostics": [
            {
                "name": "diags.zarr",
                "chunks": {"time": 12},
                "times": {"frequency": 900, "kind": "interval"},
                "variables": [
                    "storage_of_specific_humidity_path_due_to_fv3_physics",
                    "storage_of_eastward_wind_path_due_to_fv3_physics",
                    "storage_of_northward_wind_path_due_to_fv3_physics",
                    "storage_of_air_temperature_path_due_to_fv3_physics",
                    "storage_of_specific_humidity_path_due_to_emulator",
                    "storage_of_eastward_wind_path_due_to_emulator",
                    "storage_of_northward_wind_path_due_to_emulator",
                    "storage_of_air_temperature_path_due_to_emulator",
                    "water_vapor_path",
                ],
            },
            {
                "name": "diags_3d.zarr",
                "times": {"kind": "interval", "frequency": 900},
                "chunks": {"time": 12},
                "variables": [
                    "tendency_of_air_temperature_due_to_fv3_physics",
                    "tendency_of_specific_humidity_due_to_fv3_physics",
                    "tendency_of_eastward_wind_due_to_fv3_physics",
                    "tendency_of_northward_wind_due_to_fv3_physics",
                    "tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics",
                    "tendency_of_air_temperature_due_to_emulator",
                    "tendency_of_specific_humidity_due_to_emulator",
                    "tendency_of_eastward_wind_due_to_emulator",
                    "tendency_of_northward_wind_due_to_emulator",
                    "tendency_of_cloud_water_mixing_ratio_due_to_emulator",
                    "tendency_of_air_temperature_due_to_dynamics",
                    "tendency_of_specific_humidity_due_to_dynamics",
                    "tendency_of_eastward_wind_due_to_dynamics",
                    "tendency_of_northward_wind_due_to_dynamics",
                    "tendency_of_cloud_water_mixing_ratio_due_to_dynamics",
                    # input variables for emulator
                    "emulator_eastward_wind",
                    "emulator_northward_wind",
                    "emulator_air_temperature",
                    "emulator_specific_humidity",
                    "emulator_cloud_water_mixing_ratio",
                    "emulator_pressure_thickness_of_atmospheric_layer",
                    "emulator_vertical_thickness_of_atmospheric_layer",
                    "emulator_cos_zenith_angle",
                    "emulator_surface_pressure",
                    "emulator_latent_heat_flux",
                    "emulator_sensible_heat_flux",
                ],
            },
        ],
    }
