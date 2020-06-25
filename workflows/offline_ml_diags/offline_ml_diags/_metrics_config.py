from copy import deepcopy
from typing import Mapping, Sequence, Union
import xarray as xr
from vcm import safe

def fill_config_weights(
        ds: xr.Dataset, base_config: Mapping[str, Mapping[str, Sequence[str]]]
    ) -> Mapping[str, Sequence[Union[xr.DataArray, str]]]:
    """ fills in the kwargs string of weight array names with
    the data arrays themselves
    """
    config = deepcopy(base_config)
    for var, metric_config in config.items():
        for metric_name, kwargs in metric_config.items():
            if "weights" in kwargs:
                weight_vars = kwargs["weights"]
                config[var][metric_name]["weights"] = [ds[weight_var] for weight_var in weight_vars]
    return config


SCALAR_METRIC_KWARGS = {
    "column_integrated_dQ1":
        {"rmse":
            {
                "weights": ["area_weights",],
                "mean_dims": None,
            },
         "bias":
             {
                "weights": ["area_weights"],
                "mean_dims": None,
            },
        },
    "column_integrated_dQ2":
        {"rmse":
            {
                "weights": ["area_weights",],
                "mean_dims": None,
            },
         "bias":
             {
                "weights": ["area_weights",],
                "mean_dims": None,
            },
        },
    "column_integrated_Q1":
        {"rmse":
            {
                "weights": ["area_weights",],
                "mean_dims": None,
            },
         "bias":
             {
                "weights": ["area_weights",],
                "mean_dims": None,
            },
        },
    "column_integrated_Q2":
        {"rmse":
            {
                "weights": ["area_weights"],
                "mean_dims": None,
            },
         "bias":
             {
                "weights": ["area_weights",],
                "mean_dims": None,
            },
        },
    "dQ1":
         {
             "rmse":
                {
                    "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
         },
    "dQ2":
         {
             "rmse":
                {
                    "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
            },
    "Q1":
         {
             "rmse":
                {
                    "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
            },
    "Q2":
         {
             "rmse":
                {
                    "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
            }
}
