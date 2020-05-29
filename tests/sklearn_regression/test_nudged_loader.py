import pytest
import os
import xarray as xr
import numpy as np
import pandas as pd

import synth

from fv3net.regression.loaders._nudged import (
    NUDGED_TIME_DIM,
    TIME_FMT,
    _load_nudging_batches,
    _get_path_for_nudging_timescale,
    NudgedTimestepMapper,
    NudgedMapperAllSources,
)


@pytest.fixture
def xr_dataset():

    dat = np.arange(120).reshape(5, 2, 3, 4)
    ds = xr.Dataset(
        data_vars={"data": (("time", "x", "y", "z"), dat)},
        coords={
            "time": np.arange(5),
            "x": np.arange(2),
            "y": np.arange(3),
            "z": np.arange(4),
        },
    )

    return ds


# TODO session scoped
@pytest.fixture
def nudged_output_ds_dict(datadir_session):

    nudged_datasets = {}
    ntimes = 144

    nudging_tendency_path = datadir_session.join("nudging_tendencies.json")
    with open(nudging_tendency_path) as f:
        nudging_schema = synth.load(f)
    nudging_tend_ds = synth.generate(nudging_schema)
    nudging_tend_ds = _int64_to_datetime(nudging_tend_ds)
    nudged_datasets["nudging_tendencies"] = nudging_tend_ds

    # Create identically schema'd datasets to mimic before-after fv3 states
    fv3_state_schema_path = datadir_session.join(f"after_physics.json")

    with open(str(fv3_state_schema_path)) as f:
        fv3_state_schema = synth.load(f)

    datasets_sources = ["before_dynamics", "after_dynamics", "after_physics"]
    for source in datasets_sources:
        ds = synth.generate(fv3_state_schema)
        ds = _int64_to_datetime(ds)
        nudged_datasets[source] = ds.isel({NUDGED_TIME_DIM: slice(0, ntimes)})

    return nudged_datasets


def _int64_to_datetime(ds):
    time = pd.to_datetime(ds[NUDGED_TIME_DIM].values)
    time = time.round('S')
    return ds.assign_coords({NUDGED_TIME_DIM: time})


@pytest.fixture
def nudged_tstep_mapper(nudged_output_ds_dict):

    to_combine = ["nudging_tendencies", "after_physics"]
    for i, source in enumerate(to_combine):
        ds = nudged_output_ds_dict[source]

        if i == 0:
            combined_ds = ds
        else:
            combined_ds = combined_ds.merge(ds, join="inner")

    timestep_mapper = NudgedTimestepMapper(combined_ds)

    return timestep_mapper


# Note: datadir_session fixture in conftest.py
@pytest.mark.regression
def test_load_nudging_batches(nudged_tstep_mapper):

    ntimes = 90
    init_time_skip_hr = 12  # 48 15-min timesteps
    times_per_batch = 45
    num_batches = 2

    rename = {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
    }
    input_vars = ["air_temperature", "specific_humidity"]
    output_vars = ["dQ1", "dQ2"]
    data_vars = input_vars + output_vars

    # skips first 48 timesteps, only use 90 timesteps
    sequence = _load_nudging_batches(
        nudged_tstep_mapper,
        data_vars,
        num_batches=num_batches,
        num_times_in_batch=times_per_batch,
        rename_variables=rename,
        initial_time_skip_hr=init_time_skip_hr,
        n_times=ntimes,
    )

    # 14 batches requested
    assert len(sequence._args) == num_batches

    for batch in sequence:
        assert batch.sizes["sample"] == times_per_batch * 6 * 48 * 48
        for var in data_vars:
            assert var in batch


@pytest.fixture
def nudging_output_dirs(tmpdir):

    # nudging dirs which might be confusing for parser
    dirs = ["1.00", "1.5", "15"]

    nudging_dirs = {}
    for item in dirs:
        curr_dir = os.path.join(tmpdir, f"outdir-{item}h")
        os.mkdir(curr_dir)
        nudging_dirs[item] = curr_dir

    return nudging_dirs


@pytest.mark.parametrize(
    "timescale, expected_key",
    [(1, "1.00"), (1.0, "1.00"), (1.5, "1.5"), (1.500001, "1.5")],
)
def test__get_path_for_nudging_timescale(nudging_output_dirs, timescale, expected_key):

    expected_path = nudging_output_dirs[expected_key]
    result_path = _get_path_for_nudging_timescale(
        nudging_output_dirs.values(), timescale, tol=1e-5
    )
    assert result_path == expected_path


@pytest.mark.parametrize("timescale", [1.1, 1.00001])
def test__get_path_for_nudging_timescale_failure(nudging_output_dirs, timescale):
    with pytest.raises(KeyError):
        _get_path_for_nudging_timescale(nudging_output_dirs, timescale, tol=1e-5)


def test_NudgedTimestepMapper(nudged_output_ds_dict):
    single_ds = nudged_output_ds_dict["after_physics"]

    mapper = NudgedTimestepMapper(single_ds)

    assert len(mapper) == single_ds.sizes["time"]

    single_time = single_ds["time"].values[0]
    item = single_ds.isel({"time": single_time})
    time_key = pd.to_datetime(single_time).strftime(TIME_FMT)
    xr.testing.assert_equal(item, mapper[time_key])


def test_NudgedMapperAllSources(nudged_output_ds_dict):

    mapper = NudgedMapperAllSources(nudged_output_ds_dict)

    ds_len = sum([ds.sizes["time"] for ds in nudged_output_ds_dict.values()])
    assert len(mapper) == ds_len

    single_item = nudged_output_ds_dict["after_physics"].isel(time=0)
    time_key = pd.to_datetime(single_item.time.values[0]).strftime(TIME_FMT)
    item_key = ("after_physics", time_key)
    xr.testing.assert_equal(mapper[item_key], single_item)


def test_NudgedMapperAllSources_merge_sources(nudged_output_ds_dict):

    mapper = NudgedMapperAllSources(nudged_output_ds_dict)
    merged_tstep_mapper = mapper.merge_sources(["after_physics", "nudging_tendencies"])

    after_phys = mapper._nudged_ds["after_physics"]
    nudge_tend = mapper._nudged_ds["nudging_tendencies"]
    assert len(merged_tstep_mapper) == max(len(after_phys), len(nudge_tend))

    after_phys_item = after_phys[after_phys.keys()[0]]
    nudge_tend_item = nudge_tend[nudge_tend.keys()[0]]
    merged_item = merged_tstep_mapper[merged_tstep_mapper.keys()[0]]

    source_vars = list(after_phys_item.data_vars.keys())
    source_vars.extend(list(nudge_tend_item.data_vars.keys()))
    for var in source_vars:
        assert var in merged_item


def test_NudgedMapperAllSources_fail_merge(nudged_output_ds_dict):

    mapper = NudgedMapperAllSources(nudged_output_ds_dict)

    with pytest.raises(ValueError):
        mapper.merge_sources(["after_physics", "before_dynamics"])

    with pytest.raises(ValueError):
        mapper.merge_sources(
            ["nudging_tendencies", "after_physics", "before_dynamics"]
        )
