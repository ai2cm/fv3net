import xarray as xr
import vcm
from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
)
from fv3net.diagnostics.sklearn_model_performance import (
    DATASET_NAME_PREDICTION, 
    DATASET_NAME_FV3_TARGET,
    DATASET_NAME_SHIELD_HIRES,
)

STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]


def r2_global_values(ds):
    """ Calculate global R^2 for net precipitation and heating against
    target FV3 dataset and coarsened high res dataset
    
    Args:
        ds ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    r2_summary = xr.Dataset()
    for var in ["net_heating", "net_precipitation"]:
        r2_summary[f"R2_global_{var}_vs_target"] = vcm.r2_score(
            ds.sel(dataset=DATASET_NAME_FV3_TARGET)[var].stack(sample=STACK_DIMS),
            ds.sel(dataset=DATASET_NAME_PREDICTION)[var].stack(sample=STACK_DIMS),
            "sample",
        )
        r2_summary[f"R2_global_{var}_vs_hires"] = vcm.r2_score(
            ds.sel(dataset=DATASET_NAME_SHIELD_HIRES)[var].stack(sample=STACK_DIMS),
            ds.sel(dataset=DATASET_NAME_PREDICTION)[var].stack(sample=STACK_DIMS),
            "sample",
        ).values.item()
        r2_summary[f"R2_sea_{var}_vs_target"] = vcm.r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_FV3_TARGET), "sea")[var].stack(
                sample=STACK_DIMS
            ),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "sea")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
        r2_summary[f"R2_sea_{var}_vs_hires"] = vcm.r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_SHIELD_HIRES), "sea")[
                var
            ].stack(sample=STACK_DIMS),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "sea")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
        r2_summary[f"R2_land_{var}_vs_target"] = vcm.r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_FV3_TARGET), "land")[var].stack(
                sample=STACK_DIMS
            ),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "land")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
        r2_summary[f"R2_land_{var}_vs_hires"] = vcm.r2_score(
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_SHIELD_HIRES), "land")[
                var
            ].stack(sample=STACK_DIMS),
            vcm.mask_to_surface_type(ds.sel(dataset=DATASET_NAME_PREDICTION), "land")[var].stack(
                sample=STACK_DIMS
            ),
            "sample",
        ).values.item()
    return r2_summary
