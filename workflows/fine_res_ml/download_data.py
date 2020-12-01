import logging

import budget.budgets
import joblib
import loaders.mappers
import numpy as np
import vcm
from budget import config
from budget.data import open_merged

logging.basicConfig(level=logging.INFO)


def fine_res_to_zarr(url, output, seed=0):
    np.random.seed(seed)
    mapper = loaders.mappers.open_fine_resolution_budget(url=url)
    train = list(np.random.choice(list(mapper), 130, replace=False))
    template = mapper[train[0]].drop("time")
    output_mapper = vcm.ZarrMapping.from_schema(
        store=output, schema=template, dims=["time"], coords={"time": train}
    )

    def process(time):
        logging.info(f"Processing {time}")
        output_mapper[[time]] = mapper[time]

    joblib.Parallel(6)(joblib.delayed(process)(time) for time in train)
