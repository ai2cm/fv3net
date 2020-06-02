import pytest
import os
import xarray as xr
import pandas as pd
from itertools import chain

import synth

from fv3net.regression.loaders._nudged import (
    TIME_NAME,
    TIME_FMT,
    _load_nudging_batches,
    _get_path_for_nudging_timescale,
    NudgedTimestepMapper,
    NudgedStateCheckpoints,
    MergeNudged,
    GroupByCheckpoint
)

NTIMES = 144

@pytest.fixture
def nudge_tendencies(datadir_session):

    tendency_data_schema = datadir_session.join("nudging_tendencies.json")
    with open(tendency_data_schema) as f:
        schema = synth.load(f)
    nudging_tend = synth.generate(schema)
    nudging_tend = _int64_to_datetime(nudging_tend)

    return nudging_tend.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture
def general_nudge_schema(datadir_session):
    nudge_data_schema = datadir_session.join(f"after_physics.json")
    with open(nudge_data_schema) as f:
        schema = synth.load(f)

    return schema


@pytest.fixture
def general_nudge_output(general_nudge_schema):

    data = synth.generate(general_nudge_schema)
    data = _int64_to_datetime(data)

    return data.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture
def nudged_output_ds_dict(nudge_tendencies, general_nudge_schema):

    nudged_datasets = {"nudging_tendencies": nudge_tendencies}
    datasets_sources = ["before_dynamics", "after_dynamics", "after_physics"]
    for source in datasets_sources:
        ds = synth.generate(general_nudge_schema)
        ds = _int64_to_datetime(ds)
        nudged_datasets[source] = ds.isel({TIME_NAME: slice(0, NTIMES)})

    return nudged_datasets


def _int64_to_datetime(ds):
    time = pd.to_datetime(ds[TIME_NAME].values)
    time = time.round("S")
    return ds.assign_coords({TIME_NAME: time})


@pytest.fixture
def nudged_tstep_mapper(nudge_tendencies, general_nudge_output):

    combined_ds = xr.merge([nudge_tendencies, general_nudge_output], join="inner")

    timestep_mapper = NudgedTimestepMapper(combined_ds)

    return timestep_mapper


# Note: datadir_session fixture in conftest.py
@pytest.mark.regression
def test_load_nudging_batches(nudged_tstep_mapper):

    ntimes = 90
    init_time_skip_hr = 12  # 48 15-min timesteps
    times_per_batch = 14
    num_batches = 5

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

    assert len(mapper) == single_ds.sizes[TIME_NAME]

    single_time = single_ds[TIME_NAME].values[0]
    item = single_ds.sel({TIME_NAME: single_time})
    time_key = pd.to_datetime(single_time).strftime(TIME_FMT)
    xr.testing.assert_equal(item, mapper[time_key])


@pytest.fixture(params=[
    "dataset_only",
    "nudged_tstep_mapper_only",
    "mixed",
    "empty",
])
def mapper_to_ds_case(request, nudge_tendencies, general_nudge_output):

    if request.param == "dataset_only":
        sources = (nudge_tendencies, general_nudge_output)
    elif request.param == "nudged_tstep_mapper_only":
        sources = (
            NudgedTimestepMapper(nudge_tendencies),
            NudgedTimestepMapper(general_nudge_output),
        )
    elif request.param == "mixed":
        sources = (nudge_tendencies, NudgedTimestepMapper(general_nudge_output))
    elif request.param == "empty":
        sources = ()

    return sources

def test_MergeNudged__mapper_to_datasets(mapper_to_ds_case):

    datasets = MergeNudged._mapper_to_datasets(mapper_to_ds_case)
    for source in datasets:
        assert isinstance(source, xr.Dataset)


@pytest.fixture(params=["empty", "single", "no_overlap"])
def overlap_check_pass_datasets(request, nudge_tendencies, general_nudge_output):

    if request.param == "empty":
        sources = ()
    elif request.param == "single":
        sources = (nudge_tendencies,)
    elif request.param == "no_overlap":
        sources = (nudge_tendencies, general_nudge_output)

    return sources


def test_MergeNudged__check_dvar_overlap(overlap_check_pass_datasets):

    MergeNudged._check_dvar_overlap(*overlap_check_pass_datasets)


@pytest.fixture(params=["single_overlap", "all_overlap"])
def overlap_check_fail_datasets(request, nudge_tendencies, general_nudge_output):

    if request.param == "single_overlap":
        sources = (nudge_tendencies, general_nudge_output, general_nudge_output)
    elif request.param == "all_overlap":
        sources = (nudge_tendencies, nudge_tendencies, nudge_tendencies)

    return sources


def test_MergeNudged(nudge_tendencies, general_nudge_output):

    merged = MergeNudged(
        nudge_tendencies,
        NudgedTimestepMapper(general_nudge_output)
    )

    assert len(merged) == NTIMES

    item = merged[merged.keys()[0]]
    source_vars = chain(
        nudge_tendencies.data_vars.keys(),
        general_nudge_output.data_vars.keys(),
    )
    for var in source_vars:
        assert var in item


def test_MergeNudged__check_dvar_overlap_fail(overlap_check_fail_datasets):
    with pytest.raises(ValueError):
        MergeNudged._check_dvar_overlap(*overlap_check_fail_datasets)


def test_NudgedStateCheckpoints(nudged_output_ds_dict):

    mapper = NudgedStateCheckpoints(nudged_output_ds_dict)

    ds_len = sum([ds.sizes[TIME_NAME] for ds in nudged_output_ds_dict.values()])
    assert len(mapper) == ds_len

    single_item = nudged_output_ds_dict["after_physics"].isel(time=0)
    time_key = pd.to_datetime(single_item.time.values).strftime(TIME_FMT)
    item_key = ("after_physics", time_key)
    xr.testing.assert_equal(mapper[item_key], single_item)


@pytest.fixture
def checkpoint_mapping(general_nudge_output):

    item = general_nudge_output.isel(time=slice(0, 1))
    checkpoints = {
        ("before_dynamics", "20160801.001500"): item,
        ("after_dynamics", "20160801.001500"): item,
        ("after_dynamics", "20160801.003000"): item,
    }

    return checkpoints


def test_GroupByCheckpoint(checkpoint_mapping):

    test_groupby = GroupByCheckpoint(checkpoint_mapping)
    assert len(test_groupby) == 2
    assert list(test_groupby.keys()) == ["20160801.001500", "20160801.003000"]


@pytest.mark.parametrize(
    "time_key, expected_size",
    [("20160801.001500", 2), ("20160801.003000", 1)]
)
def test_GroupByCheckpoint_items(time_key, expected_size, checkpoint_mapping):
    
    test_groupby = GroupByCheckpoint(checkpoint_mapping)
    
    item = test_groupby[time_key]
    assert "checkpoint" in item.dims
    assert item.sizes["checkpoint"] == expected_size
