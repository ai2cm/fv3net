"""Prepares mock data schema for the fine-res data

To run this script, first get the path to the nudging training data used by the
test. This can be seen be examining the logs of

    pytest tests/test__nudged.py

for any errors. If there are no errors, you don't need to update the data!

Once you have temporary the nudging path copy paste it into "nudge_url" below.
"""
# coding: utf-8
from loaders.mappers._hybrid import open_zarr_maybe_consolidated
from datetime import timedelta
import synth
import tempfile
import os

fine_url: str = "gs://vcm-ml-experiments/default/2021-04-27/2020-05-27-40-day-X-SHiELD-simulation/fine-res-budget.zarr"  # noqa
# path to interrupted nudge_to_fine_data from a failed test
nudge_url = "/tmp/pytest-of-noahb/pytest-11/pytest_data0/nudge_to_fine_data"


fine = open_zarr_maybe_consolidated(fine_url)

nudge_physics_tendencies = open_zarr_maybe_consolidated(
    nudge_url + "/physics_tendencies.zarr",
)
nudge_state = open_zarr_maybe_consolidated(nudge_url + "/state_after_timestep.zarr")
nudge_tends = open_zarr_maybe_consolidated(nudge_url + "/nudging_tendencies.zarr")

fine_time = nudge_state.time + timedelta(minutes=7, seconds=30)
fine_subsampled = (
    fine.isel(time=slice(0, len(fine_time)))
    .assign_coords(time=fine_time)
    .isel(grid_xt=slice(0, 8), grid_yt=slice(0, 8), pfull=slice(0, 19))
)

with tempfile.TemporaryDirectory() as dir_:
    fine_subsampled.to_zarr(os.path.join(dir_, "fine_res.zarr"))
    schema = synth.read_directory_schema(dir_)

synth.dump_directory_schema_to_disk(schema, "tests/test__nudged/fine_res")
