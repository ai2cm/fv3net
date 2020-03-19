from typing import List
import yaml
import sys
import fv3config
from fv3net.pipelines.kube_jobs import utils
from copy import deepcopy
import fsspec
import os
import logging
import xarray as xr
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTDIR = "/tmp/blah"
RUNFILE = "/workflows/one_step_jobs/runfile.py"


def _get_initial_condition_assets(input_url: str, timestep: str) -> List[dict]:
    """
    Get list of assets representing initial conditions for this timestep to pipeline.
    """
    initial_condition_assets = utils.update_tiled_asset_names(
        source_url=input_url,
        source_filename="{timestep}/{timestep}.{category}.tile{tile}.nc",
        target_url="INPUT",
        target_filename="{category}.tile{tile}.nc",
        timestep=timestep,
    )

    return initial_condition_assets


def _current_date_from_timestep(timestep: str) -> List[int]:
    """Return timestep in the format required by fv3gfs namelist"""
    year = int(timestep[:4])
    month = int(timestep[4:6])
    day = int(timestep[6:8])
    hour = int(timestep[9:11])
    minute = int(timestep[11:13])
    second = int(timestep[13:15])
    return [year, month, day, hour, minute, second]


def _assoc_initial_conditions(
    base_config: dict,
    input_url: str,
    timestep: str,
) -> dict:
    config = deepcopy(base_config)
    config["experiment_name"] = timestep
    config["initial_conditions"] += _get_initial_condition_assets(
        input_url, timestep
    )
    config["namelist"]["coupler_nml"].update(
        {
            "current_date": _current_date_from_timestep(timestep),
            "force_date_from_namelist": True,
        }
    )

    return config


def post_process(outdir, store_url, index):
    logger.info("Post processing model outputs")
    begin = xr.open_zarr(f"{outdir}/begin_physics.zarr")
    before = xr.open_zarr(f"{outdir}/before_physics.zarr")
    after = xr.open_zarr(f"{outdir}/after_physics.zarr")

    # make the time dims consistent
    time = begin.time
    before = before.drop('time')
    after = after.drop('time')
    begin = begin.drop('time')

    # concat data
    dt = np.timedelta64(15, 'm')
    time = np.arange(len(time)) * dt
    ds = xr.concat([begin, before, after], dim='step').assign_coords(step=['begin', 'after_dynamics', 'after_physics'], time=time)
    ds = ds.rename({'time': 'lead_time'})

    # put in storage
    # this object must be initialized
    mapper = fsspec.get_mapper(store_url)
    group = zarr.open_group(mapper, mode='a')
    for variable in group:
        logger.info(f"Writing {variable} to {group}")
        dims = group[variable].attrs['_ARRAY_DIMENSIONS'][1:]
        group[variable][index] = np.asarray(ds[variable].transpose(*dims))



if __name__ == "__main__":
    input_url, output_url, timestep, index = sys.argv[1:]

    with fsspec.open(os.path.join(output_url, "fv3config.yml")) as f:
        base_config = yaml.safe_load(f)

    config = _assoc_initial_conditions(base_config, input_url, timestep)

    logger.info("Dumping yaml to remote")
    with fsspec.open(os.path.join(output_url, "fv3config", f"{timestep}.yml")) as f:
        yaml.safe_dump(config, f)

    logger.info("Running FV3")
    fv3config.run_native(config, outdir=OUTDIR, runfile=RUNFILE)
    post_process(OUTDIR, store_url=f"{output_url}/big.zarr", index=int(index))
