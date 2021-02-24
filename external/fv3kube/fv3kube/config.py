import os
import datetime
from pathlib import Path
from typing import Any, Sequence, Mapping, Optional
import fsspec
import yaml
import fv3config


# Map for different base fv3config dictionaries
PWD = Path(os.path.abspath(__file__)).parent
DEFAULT_BASE_VERSION = "v0.3"  # for backward compatibility
BASE_FV3CONFIG_BY_VERSION = {
    "v0.2": os.path.join(PWD, "base_yamls/v0.2/fv3config.yml"),
    "v0.3": os.path.join(PWD, "base_yamls/v0.3/fv3config.yml"),
    "v0.4": os.path.join(PWD, "base_yamls/v0.4/fv3config.yml"),
    "v0.5": os.path.join(PWD, "base_yamls/v0.5/fv3config.yml"),
    "v0.6": os.path.join(PWD, "base_yamls/v0.6/fv3config.yml"),
}

TILE_COORDS_FILENAMES = range(1, 7)  # tile numbering in model output filenames
RESTART_CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer.res", "fv_srf_wnd.res"]
FV_CORE_ASSET = fv3config.get_asset_dict(
    "gs://vcm-fv3config/data/initial_conditions/fv_core_79_levels/v1.0/",
    "fv_core.res.nc",
    target_location="INPUT",
)
FV3Config = Mapping[str, Any]


def _merge_once(source, update):
    """Recursively update a mapping with new values.

    Args:
        source: Mapping to be updated.
        update: Mapping whose key-value pairs will update those in source.
            Key-value pairs will be inserted for keys in update that do not exist
            in source.

    Returns:
        Recursively updated mapping.
    """
    for key in update:
        if key in ["patch_files", "diagnostics"]:
            source.setdefault(key, []).extend(update[key])
        elif (
            key in source
            and isinstance(source[key], Mapping)
            and isinstance(update[key], Mapping)
        ):
            _merge_once(source[key], update[key])
        else:
            source[key] = update[key]
    return source


def merge_fv3config_overlays(*mappings) -> Mapping:
    """Recursive merge dictionaries updating from left to right.

    For example, the rightmost mapping will override the proceeding ones. """
    out, rest = mappings[0], mappings[1:]
    for mapping in rest:
        out = _merge_once(out, mapping)
    return out


def get_base_fv3config(version_key: Optional[str] = None) -> FV3Config:
    """
    Get base configuration dictionary labeled by version_key.
    """
    version_key = version_key or DEFAULT_BASE_VERSION
    config_path = BASE_FV3CONFIG_BY_VERSION[version_key]
    with fsspec.open(config_path) as f:
        base_yaml = yaml.safe_load(f)

    return base_yaml


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


def get_full_config(
    config_update: FV3Config, ic_url: str, ic_timestep: str
) -> FV3Config:
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
    return merge_fv3config_overlays(
        get_base_fv3config(base_version),
        c48_initial_conditions_overlay(ic_url, ic_timestep),
        config_update,
    )


def c48_initial_conditions_overlay(url: str, timestep: str) -> Mapping:
    """An overlay containing initial conditions namelist settings
    """
    TIME_FMT = "%Y%m%d.%H%M%S"
    time = datetime.datetime.strptime(timestep, TIME_FMT)
    time_list = [time.year, time.month, time.day, time.hour, time.minute, time.second]

    overlay = {}
    overlay["initial_conditions"] = update_tiled_asset_names(
        source_url=os.path.join(url, timestep),
        source_filename="{timestep}.{category}.tile{tile}.nc",
        target_url="INPUT",
        target_filename="{category}.tile{tile}.nc",
        timestep=timestep,
    )
    overlay["initial_conditions"].append(FV_CORE_ASSET)
    overlay["namelist"] = {}
    overlay["namelist"]["coupler_nml"] = {
        "current_date": time_list,
        "force_date_from_namelist": True,
    }

    return overlay
