from typing import Mapping, Sequence
from constants import VERIFICATION_CATALOG_ENTRIES


def get_verification_entries(name: str) -> Mapping[str, Sequence]:
    """Given name, return fv3net catalog keys for verification dycore and
    physics data."""
    if name not in VERIFICATION_CATALOG_ENTRIES:
        raise ValueError(
            f"Invalid verification option. Got {name}, valid options are "
            f"{set(VERIFICATION_CATALOG_ENTRIES.keys())}."
        )
    return VERIFICATION_CATALOG_ENTRIES[name]
