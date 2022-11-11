"""Script for updating the regression test data for the restart coarsening
tests.  We cannot use a traditional checksum-based system, since results of
mappm are not bit for bit reproducible on a VM versus in CircleCI.  Instead we
store reference coarsened data, which we compare results to in an approximate
way.  This script will overwrite the existing reference coarse data for each
test case.
"""
import logging

from test_coarsen_restarts import (
    FACTOR,
    REGRESSION_TESTS,
    open_fine_restarts,
    open_fine_grid,
    reference_store,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    restarts = open_fine_restarts()
    grid = open_fine_grid()

    for tag, (func, kwargs) in REGRESSION_TESTS.items():
        result = func(FACTOR, grid, restarts, **kwargs)
        for category, ds in result.items():
            store = reference_store(tag, category)
            logging.info(f"Writing new regression data to {store}")
            ds.load().to_zarr(store, mode="w")
