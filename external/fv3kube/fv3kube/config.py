import os
from pathlib import Path
from typing import Any, Sequence, Mapping
import fsspec
import yaml

import fv3config
import vcm
from vcm.cloud import get_protocol, get_fs


# Map for different base fv3config dictionaries
PWD = Path(os.path.abspath(__file__)).parent
DEFAULT_BASE_VERSION = "v0.3"  # for backward compatibility
BASE_FV3CONFIG_BY_VERSION = {
    "v0.2": os.path.join(PWD, "base_yamls/v0.2/fv3config.yml"),
    "v0.3": os.path.join(PWD, "base_yamls/v0.3/fv3config.yml"),
    "v0.4": os.path.join(PWD, "base_yamls/v0.4/fv3config.yml"),
}

TILE_COORDS_FILENAMES = range(1, 7)  # tile numbering in model output filenames
RESTART_CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer.res", "fv_srf_wnd.res"]
FV_CORE_ASSET = fv3config.get_asset_dict(
    "gs://vcm-fv3config/data/initial_conditions/fv_core_79_levels/v1.0/",
    "fv_core.res.nc",
    target_location="INPUT",
)
FV3Config = Mapping[str, Any]


def get_base_fv3config(version_key: str) -> FV3Config:
    """
    Get base configuration dictionary labeled by version_key.
    """
    config_path = BASE_FV3CONFIG_BY_VERSION[version_key]
    with fsspec.open(config_path) as f:
        base_yaml = yaml.safe_load(f)

    return base_yaml


def transfer_local_to_remote(path: str, remote_url: str) -> str:
    """
    Transfer a local file to a remote path and return that remote path.
    If path is already remote, this does nothing.
    """
    if get_protocol(path) == "file":
        remote_path = os.path.join(remote_url, os.path.basename(path))
        get_fs(remote_url).put(path, remote_path)
        path = remote_path
    return path


def update_tiled_asset_names(
    source_url: str,
    source_filename: str,
    target_url: str,
    target_filename: str,
    **kwargs,
) -> Sequence[Mapping[str, str]]:

    """
    Update tile-based fv3config assets with new names.  Uses format to update
    any data in filename strings with category, tile, and provided keyword
    arguments.

    Filename strings should include any specified variable name inserts to
    be updated with a format. E.g., "{timestep}.{category}.tile{tile}.nc"
    """
    assets = [
        fv3config.get_asset_dict(
            source_url,
            source_filename.format(category=category, tile=tile, **kwargs),
            target_location=target_url,
            target_name=target_filename.format(category=category, tile=tile, **kwargs),
        )
        for category in RESTART_CATEGORIES
        for tile in TILE_COORDS_FILENAMES
    ]

    return assets


def get_full_config(config_update: FV3Config, ic_url: str, ic_timestep: str) -> FV3Config:
    """Given config_update return full fv3config object pointing to initial conditions
    at {ic_url}/{ic_timestep}. Initial condition filenames assumed to include prepended
    timestamp.

    Args:
        config_update: fv3config update object containing parameters different from
            the base config specified by its "base_version" argument
        ic_url: path to directory containing all initial conditions
        ic_timestep: timestamp of desired initial condition

    Returns:
        fv3config Mapping
    """
    base_version = config_update.get("base_version", DEFAULT_BASE_VERSION)
    base_config = get_base_fv3config(base_version)
    full_config = vcm.update_nested_dict(base_config, config_update)
    full_config["initial_conditions"] = update_tiled_asset_names(
        source_url=os.path.join(ic_url, ic_timestep),
        source_filename="{timestep}.{category}.tile{tile}.nc",
        target_url="INPUT",
        target_filename="{category}.tile{tile}.nc",
        timestep=ic_timestep,
    )
    full_config["initial_conditions"].append(FV_CORE_ASSET)
    full_config["namelist"]["coupler_nml"].update(
        {
            "current_date": vcm.parse_current_date_from_str(ic_timestep),
            "force_date_from_namelist": True,
        }
    )
    return full_config
