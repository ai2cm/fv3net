import synth
import xarray as xr
import numpy as np
import pytest
import os
import logging
import diagnostics_utils as utils
import intake
from fv3net.regression import loaders

logger = logging.getLogger(__name__)

DOMAINS = ["land", "sea", "global"]
OUTPUT_NC_NAME = "diagnostics.nc"


@pytest.mark.regression()
def test_compute_diags(datadir):

    # load schema and generate input datasets, each with two timesteps

    output_dir = str(datadir.join("out"))

    # one steps
    one_step_dir = "one_step"
    with open(str(datadir.join("one_step.json"))) as f:
        one_step_schema = synth.load(f)
    one_step_dataset = synth.generate(one_step_schema)
    one_step_dataset_1 = one_step_dataset.assign_coords(
        {"initial_time": ["20160901.000000"]}
    )
    one_step_dataset_2 = one_step_dataset.assign_coords(
        {"initial_time": ["20160901.001500"]}
    )
    one_step_dataset_1.to_zarr(
        os.path.join(output_dir, one_step_dir, "20160901.000000.zarr"),
        consolidated=True,
    )
    one_step_dataset_2.to_zarr(
        os.path.join(output_dir, one_step_dir, "20160901.001500.zarr"),
        consolidated=True,
    )

    # nudging
    nudging_dir = "outdir-3h"
    nudging_after_dynamics_zarrpath = os.path.join(
        output_dir, nudging_dir, "after_dynamics.zarr"
    )
    with open(str(datadir.join("after_dynamics.json"))) as f:
        nudging_after_dynamics_schema = synth.load(f)
    nudging_after_dynamics_dataset = synth.generate(
        nudging_after_dynamics_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64("2016-09-01T00:00:00"),
                np.datetime64("2016-09-01T00:15:00"),
            ]
        }
    )
    nudging_after_dynamics_dataset.to_zarr(
        nudging_after_dynamics_zarrpath, consolidated=True
    )

    nudging_after_physics_zarrpath = os.path.join(
        output_dir, nudging_dir, "after_physics.zarr"
    )
    with open(str(datadir.join("after_physics.json"))) as f:
        nudging_after_physics_schema = synth.load(f)
    nudging_after_physics_dataset = synth.generate(
        nudging_after_physics_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64("2016-09-01T00:00:00"),
                np.datetime64("2016-09-01T00:15:00"),
            ]
        }
    )
    nudging_after_physics_dataset.to_zarr(
        nudging_after_physics_zarrpath, consolidated=True
    )

    nudging_tendencies_zarrpath = os.path.join(
        output_dir, nudging_dir, "nudging_tendencies.zarr"
    )
    with open(str(datadir.join("nudging_tendencies.json"))) as f:
        nudging_tendencies_schema = synth.load(f)
    nudging_tendencies_dataset = synth.generate(
        nudging_tendencies_schema
    ).assign_coords(
        {
            "time": [
                np.datetime64("2016-09-01T00:00:00"),
                np.datetime64("2016-09-01T00:15:00"),
            ]
        }
    )
    nudging_tendencies_dataset.to_zarr(nudging_tendencies_zarrpath, consolidated=True)

    # fine res
    fine_res_zarrpath = os.path.join(
        output_dir, "fine_res_budget", "fine_res_budget.zarr"
    )
    with open(str(datadir.join("fine_res_budget.json"))) as f:
        fine_res_budget_schema = synth.load(f)
    fine_res_budget_dataset = synth.generate(fine_res_budget_schema)
    fine_res_budget_dataset_1 = fine_res_budget_dataset.assign_coords(
        {"time": ["20160901.000000"]}
    )
    fine_res_budget_dataset_2 = fine_res_budget_dataset.assign_coords(
        {"time": ["20160901.001500"]}
    )
    fine_res_budget_dataset = xr.concat(
        [fine_res_budget_dataset_1, fine_res_budget_dataset_2], dim="time"
    )
    fine_res_budget_dataset.to_zarr(fine_res_zarrpath, consolidated=True)

    # load the grid

    cat = intake.open_catalog("catalog.yml")
    grid = cat["grid/c48"].to_dask()
    grid = grid.drop(labels=["y_interface", "y", "x_interface", "x"])
    surface_type = cat["grid/c48/land_sea_mask"].to_dask()
    surface_type = surface_type.drop(labels=["y", "x"])
    grid = grid.merge(surface_type)

    # compute the diagnostics for each source

    variable_names = [
        "dQ1",
        "dQ2",
        "pQ1",
        "pQ2",
        "pressure_thickness_of_atmospheric_layer",
    ]

    diagnostic_datasets = {}
    num_batches = 2
    timesteps_per_batch = 1

    # one step
    ds_batches_one_step = loaders.diagnostic_sequence_from_mapper(
        os.path.join(output_dir, one_step_dir),
        variable_names,
        num_batches=num_batches,
        timesteps_per_batch=timesteps_per_batch,
        mapping_function="open_one_step",
    )
    ds_diagnostic_one_step = utils.reduce_to_diagnostic(
        ds_batches_one_step, grid, domains=DOMAINS
    )
    diagnostic_datasets["one_step_tendencies"] = ds_diagnostic_one_step

    # nudged
    nudged_mapping_kwargs = {
        "nudging_timescale_hr": 3,
        "open_merged_nudged_kwargs": {
            "rename_vars": {
                "air_temperature_tendency_due_to_nudging": "dQ1",
                "specific_humidity_tendency_due_to_nudging": "dQ2",
            }
        },
        "open_checkpoints_kwargs": {
            "checkpoint_files": ("after_dynamics.zarr", "after_physics.zarr")
        },
    }

    ds_batches_nudged = loaders.diagnostic_sequence_from_mapper(
        output_dir,
        variable_names,
        num_batches=num_batches,
        timesteps_per_batch=timesteps_per_batch,
        mapping_function="open_nudged_tendencies",
        mapping_kwargs=nudged_mapping_kwargs,
    )
    ds_diagnostic_nudged = utils.reduce_to_diagnostic(
        ds_batches_nudged, grid, domains=DOMAINS
    )
    diagnostic_datasets["nudged_tendencies"] = ds_diagnostic_nudged

    # fine res, open without batch loader since synth format isn't easily
    # compatible with actual data

    rename_variables = {
        "delp": "pressure_thickness_of_atmospheric_layer",
        "pfull": "z",
        "grid_xt": "x",
        "grid_yt": "y",
    }
    ds_batches_fine_res = [
        xr.open_zarr(fine_res_zarrpath).isel(time=0).rename(rename_variables),
        xr.open_zarr(fine_res_zarrpath).isel(time=1).rename(rename_variables),
    ]
    ds_diagnostic_fine_res = utils.reduce_to_diagnostic(
        ds_batches_fine_res, grid, domains=DOMAINS
    )
    diagnostic_datasets["fine_res_apparent_sources"] = ds_diagnostic_fine_res

    diagnostics_all = xr.concat(
        [
            dataset.expand_dims({"data_source": [dataset_name]})
            for dataset_name, dataset in diagnostic_datasets.items()
        ],
        dim="data_source",
    ).load()

    diags_output_schema_raw = synth.read_schema_from_dataset(diagnostics_all)
    # TODO standardize schema encoding in synth to avoid the casting that makes
    # the following line necessary (arrays vs lists)
    diags_output_schema = synth.loads(synth.dumps(diags_output_schema_raw))

    # test against reference

    with open(str(datadir.join("diags_reference.json"))) as f:
        reference_output_schema = synth.load(f)

    assert reference_output_schema == diags_output_schema
