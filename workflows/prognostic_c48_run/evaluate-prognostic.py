from pathlib import Path
import tempfile

import wandb
import xarray
import subprocess
import yaml
import os

from runtime.emulator.prognostic import compute_metrics, open_run

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore


def _get_config():
    return {
        "base_version": "v0.5",
        "forcing": "gs://vcm-fv3config/data/base_forcing/v1.1/",
        "online_emulator": {
            "checkpoint": "dummy/path",
            "train": False,
            "online": True,
            "ignore_humidity_below": None,
        },
        "namelist": {
            "coupler_nml": {"days": 0, "hours": 3, "minutes": 0, "seconds": 0},
            "diag_manager_nml": {"flush_nc_files": True},
            "fv_core_nml": {
                "do_sat_adj": False,
                "warm_start": True,
                "external_ic": False,
                "external_eta": False,
                "nggps_ic": False,
                "make_nh": False,
                "mountain": True,
                "nwat": 2,
            },
            "gfdl_cloud_microphysics_nml": {"fast_sat_adj": False},
            "gfs_physics_nml": {
                "fhzero": 0.25,
                "satmedmf": False,
                "hybedmf": True,
                "imp_physics": 99,
                "ncld": 1,
            },
        },
        "fortran_diagnostics": [
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
                    {
                        "module_name": "dynamics",
                        "field_name": "w50",
                        "output_name": "w50",
                    },
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
                    {
                        "module_name": "dynamics",
                        "field_name": "tq",
                        "output_name": "PWAT",
                    },
                    {
                        "module_name": "dynamics",
                        "field_name": "lw",
                        "output_name": "VIL",
                    },
                    {
                        "module_name": "dynamics",
                        "field_name": "iw",
                        "output_name": "iw",
                    },
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
        ],
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
                ],
            },
        ],
    }


def log_vertical_metrics(key, metrics: xarray.Dataset):
    df = metrics.to_dataframe().reset_index()
    df["time"] = df.time.apply(lambda x: x.isoformat())
    wandb.log({key: wandb.Table(dataframe=df)})


def log_summary_metrics(label: str, mean: xarray.Dataset):
    for key in mean:
        wandb.summary[label + "/" + str(key)] = float(mean[key])


def run(config, path, ic_url, ic):

    with tempfile.NamedTemporaryFile("w") as f, tempfile.NamedTemporaryFile(
        "w"
    ) as user_config:
        yaml.safe_dump(config, user_config)
        subprocess.check_call(
            ["prepare-config", user_config.name, ic_url, ic], stdout=f,
        )
        subprocess.check_call(["runfv3", "create", path, f.name, "sklearn_runfile.py"])
    subprocess.check_call(["runfv3", "append", path])


def evaluate(path):
    ds = open_run(path)
    metrics = compute_metrics(ds)
    log_vertical_metrics("vertical_metrics", metrics)
    log_summary_metrics("mean", metrics.mean())
    wandb.finish()


# boiler plate to get allow hydra to modify the default fv3config dictionary
cs = ConfigStore.instance()
CONFIG_NAME = "config"
cs.store(
    name=CONFIG_NAME,
    node=dict(
        fv3=_get_config(),
        ic_url=(
            "gs://vcm-ml-experiments/andrep/2021-05-28/spunup-c48-simple-phys-hybrid-edmf"  # noqa
        ),
        ic="20160802.000000",
        artifact_id="doesn't matter",
    ),
)


@hydra.main(config_path=None, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    job = wandb.init(entity="ai2cm", project="emulator-noah", job_type="prognostic-run")
    wandb.config.update(OmegaConf.to_container(cfg))

    path = Path("/data/prognostic-runs") / str(job.id)
    artifact = wandb.use_artifact(cfg.artifact_id)
    artifact_path = os.path.abspath(artifact.download())

    config = cfg.fv3
    config["online_emulator"]["checkpoint"] = artifact_path
    run(OmegaConf.to_container(config), path, ic_url=cfg.ic_url, ic=cfg.ic)
    evaluate(path)


if __name__ == "__main__":
    main()
