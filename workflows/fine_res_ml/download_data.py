from budget.data import open_merged
import budget.budgets
from budget import config
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

output = "~/dev/data/2020-11-25-fine-res.zarr"

data = open_merged(
    config.restart_url, config.physics_url, config.gfsphysics_url, config.area_url
)
train = np.random.choice(data.time, 130, replace=False)

coarse = budget.budgets.compute_recoarsened_budget_inputs(
    data.sel(time=train), factor=config.factor, first_moments=config.VARIABLES_TO_AVERAGE
)
coarse.chunk({"time": 1, "grid_xt": -1, "grid_yt": -1, "tile": 1}).to_zarr(output, mode='w')


