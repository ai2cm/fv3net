"""
This module is for functions that select subsets of the data
"""
import warnings


def mask_to_surface_type(ds, surface_type):
    """
    Args:
        ds: xarray dataset, must have variable slmsk
        surface_type: one of ['sea', 'land', 'seaice']
    Returns:
        input dataset masked to the surface_type specified
    """
    if surface_type is None:
        warnings.warn("surface_type provided as None: no mask applied.")
        return ds
    elif surface_type not in ["sea", "land", "seaice"]:
        raise ValueError("Must mask to surface_type in ['sea', 'land', 'seaice'].")
    surface_type_codes = {"sea": 0, "land": 1, "seaice": 2}
    mask = ds.slmsk.astype(int) == surface_type_codes[surface_type]
    ds_masked = ds.where(mask)
    return ds_masked
