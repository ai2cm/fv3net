import os
from pathlib import Path
from typing import Any, Mapping, Optional
import fsspec
import yaml


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


def get_full_config(config_update: FV3Config) -> FV3Config:
    """Given config_update return full fv3config object.
    Args:
        config_update: fv3config update object containing parameters different from
            the base config specified by its "base_version" argument
    Returns:
        fv3config Mapping
    """
    base_version = config_update.get("base_version", DEFAULT_BASE_VERSION)
    return merge_fv3config_overlays(get_base_fv3config(base_version), config_update,)
