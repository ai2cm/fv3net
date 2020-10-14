from typing import Mapping
import intake


def get_verification_entries(name: str, catalog: intake.Catalog) -> Mapping[str, str]:
    """Given simulation name, return fv3net catalog keys for related c48 dycore and
    physics data."""
    entries = {"physics": [], "dycore": []}
    for item in catalog:
        metadata = catalog[item].metadata
        item_simulation = metadata.get("simulation", None)
        item_grid = metadata.get("grid", None)
        item_category = metadata.get("category", None)

        if item_simulation == name and item_grid == "c48":
            if item_category is not None:
                entries[item_category].append(item)

    if len(entries["physics"]) == 0:
        raise ValueError(
            f"No c48 physics data found in catalog for simulation {name}. Check "
            "verification tag."
        )
    if len(entries["dycore"]) == 0:
        raise ValueError(
            f"No c48 dycore data found in catalog for simulation {name}. Check "
            "verification tag."
        )

    return entries
