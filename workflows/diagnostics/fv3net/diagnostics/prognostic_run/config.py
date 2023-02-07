from typing import Mapping, List
import intake


def get_verification_entries(
    name: str, catalog: intake.catalog.Catalog, evaluation_grid="c48"
) -> Mapping[str, List[str]]:
    """Given simulation name, return catalog keys for c48 dycore and physics data.

    Args:
        name: Simulation to use for verification.
        catalog: Catalog to search for verification data.
        evaluation_grid: Grid upon which to compute diagnostics

    Returns:
        Mapping from category name ('physics', 'dycore', or '3d') to sequence
        of catalog keys representing given diagnostics for specified simulation.
    """
    entries: Mapping[str, List[str]] = {"2d": [], "3d": []}
    for item in catalog:
        metadata = catalog[item].metadata
        item_simulation = metadata.get("simulation", None)
        item_grid = metadata.get("grid", None)
        item_category = metadata.get("category", None)

        if item_simulation == name and item_grid == evaluation_grid:
            if item_category is not None:
                entries[item_category].append(item)

    if len(entries["2d"]) == 0:
        raise ValueError(
            f"No {evaluation_grid} 2d diagnostics found in catalog for "
            f"simulation {name}."
        )

    return entries
