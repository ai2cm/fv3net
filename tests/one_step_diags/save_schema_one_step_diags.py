import sys
import fsspec
import logging
import synth
import xarray as xr

logging.basicConfig(level=logging.INFO)


url = "gs://vcm-ml-scratch/brianh/one-step-diags-testing/deep-off-298d69/one_step_diag_data.nc"  # noqa
with fsspec.open(url, mode="rb") as f:
    dataset = xr.open_dataset(f).load()
schema = synth.read_schema_from_dataset(dataset)
synth.dump(schema, sys.stdout)
