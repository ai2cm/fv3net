from typing import Mapping, Sequence

VERIFICATION_CATALOG_ENTRIES = {
    "nudged_shield_40day": {
        "physics": ("40day_c48_gfsphysics_15min_may2020",),
        "dycore": ("40day_c48_atmos_8xdaily_may2020",),
    },
    "nudged_c48_fv3gfs_2016": {
        "dycore": ("2016_c48_nudged_fv3gfs_dycore_output",),
        "physics": ("2016_c48_nudged_fv3gfs_physics_output",),
    },
}


def get_verification_entries(name: str) -> Mapping[str, Sequence]:
    """Given name, return fv3net catalog keys for verification dycore and
    physics data."""
    if name not in VERIFICATION_CATALOG_ENTRIES:
        raise ValueError(
            f"Invalid verification option. Got {name}, valid options are "
            f"{set(VERIFICATION_CATALOG_ENTRIES.keys())}."
        )
    return VERIFICATION_CATALOG_ENTRIES[name]
