#!/bin/python3
import xarray as xr
import sys


if __name__ == "__main__":
    path = sys.argv[1]
    ds = xr.open_dataset(path)
    variables = ds.data_vars
    print(",".join(variables))
