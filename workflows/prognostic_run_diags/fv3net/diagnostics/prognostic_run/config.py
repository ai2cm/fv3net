from typing import Mapping, Sequence
import intake


def get_verification_entries(
    name: str, catalog: intake.Catalog
) -> Mapping[str, Sequence[str]]:
    """Given simulation name, return catalog keys for c48 dycore and physics data.
    
    Args:
        name: Simulation to use for verification.
        catalog: Catalog to search for verification data.
        
    Returns:
        Mapping from category name ('physics' or 'dycore') to sequence of catalog keys
        representing given diagnostics for specified simulation.
    """
    entries = {"physics": [], "dycore": [], "3d": []}
    for item in catalog:
        metadata = catalog[item].metadata
        item_simulation = metadata.get("simulation", None)
        item_grid = metadata.get("grid", None)
        item_category = metadata.get("category", None)

        if item_simulation == name and item_grid == "c48":
            if item_category is not None:
                entries[item_category].append(item)

    if len(entries["physics"]) == 0:
        raise ValueError(f"No c48 physics data found in catalog for simulation {name}.")
    if len(entries["dycore"]) == 0:
        raise ValueError(f"No c48 dycore data found in catalog for simulation {name}.")

    return entries
