import pytest
import os
import xarray as xr
import numpy as np
import pandas as pd
from itertools import chain
import synth
from vcm import safe
from loaders import TIME_NAME, TIME_FMT
from loaders.mappers import LongRunMapper
from loaders.mappers._nudged import (
    NudgedStateCheckpoints,
    MergeNudged,
    GroupByTime,
    SubsetTimes,
    NudgedFullTendencies,
    NudgeToObsState,
    NudgeToObsFullTendencies,
    open_merged_nudged,
    open_merged_nudge_to_obs,
    _open_nudging_checkpoints,
    open_merged_nudged_full_tendencies,
    open_merged_nudge_to_obs_full_tendencies,
)

NTIMES = 12


@pytest.fixture(scope="module")
def nudge_tendencies(datadir_module):

    tendency_data_schema = datadir_module.join("nudging_tendencies.json")
    with open(tendency_data_schema) as f:
        schema = synth.load(f)
    nudging_tend = synth.generate(schema)
    nudging_tend = _int64_to_datetime(nudging_tend)

    return nudging_tend.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def general_nudge_schema(datadir_module):
    nudge_data_schema = datadir_module.join(f"after_physics.json")
    with open(nudge_data_schema) as f:
        schema = synth.load(f)

    return schema


@pytest.fixture(scope="module")
def general_nudge_output(general_nudge_schema):

    data = synth.generate(general_nudge_schema)
    data = _int64_to_datetime(data)

    return data.isel({TIME_NAME: slice(0, NTIMES)})


@pytest.fixture(scope="module")
def nudged_checkpoints(general_nudge_schema):

    nudged_datasets = {}
    datasets_sources = [
        "before_dynamics",
        "after_dynamics",
        "after_physics",
        "after_nudging",
    ]
    for source in datasets_sources:
        ds = synth.generate(general_nudge_schema)
        ds = _int64_to_datetime(ds)
        nudged_datasets[source] = ds.isel({TIME_NAME: slice(0, NTIMES)})

    return nudged_datasets


@pytest.fixture(scope="module")
def nudged_data_dir(datadir_module, nudged_checkpoints, nudge_tendencies):

    all_data = dict(**nudged_checkpoints)
    all_data.update({"nudging_tendencies": nudge_tendencies})

    for filestem, ds in all_data.items():
        filepath = os.path.join(datadir_module, f"{filestem}.zarr")
        ds.to_zarr(filepath)

    return str(datadir_module)


def _int64_to_datetime(ds):
    time = pd.to_datetime(ds[TIME_NAME].values)
    time = time.round("S")
    return ds.assign_coords({TIME_NAME: time})


@pytest.fixture(scope="module")
def nudged_tstep_mapper(nudge_tendencies, general_nudge_output):

    combined_ds = xr.merge([nudge_tendencies, general_nudge_output], join="inner")

    timestep_mapper = LongRunMapper(combined_ds)

    return timestep_mapper


@pytest.fixture(params=["dataset_only", "nudged_tstep_mapper_only", "mixed"])
def mapper_to_ds_case(request, nudge_tendencies, general_nudge_output):

    if request.param == "dataset_only":
        sources = (nudge_tendencies, general_nudge_output)
    elif request.param == "nudged_tstep_mapper_only":
        sources = (
            LongRunMapper(nudge_tendencies),
            LongRunMapper(general_nudge_output),
        )
    elif request.param == "mixed":
        sources = (nudge_tendencies, LongRunMapper(general_nudge_output))

    return sources


def test_MergeNudged__mapper_to_datasets(mapper_to_ds_case):

    datasets = MergeNudged._mapper_to_datasets(mapper_to_ds_case)
    for source in datasets:
        assert isinstance(source, xr.Dataset)


@pytest.mark.parametrize("init_args", [(), ("single_arg",)])
def test_MergeNudged__fail_on_empty_sources(init_args):

    with pytest.raises(TypeError):
        MergeNudged(*init_args)


def test_MergeNudged__check_dvar_overlap(nudge_tendencies, general_nudge_output):

    MergeNudged._check_dvar_overlap(*(nudge_tendencies, general_nudge_output))


@pytest.fixture(params=["single_overlap", "all_overlap"])
def overlap_check_fail_datasets(request, nudge_tendencies, general_nudge_output):

    if request.param == "single_overlap":
        sources = (nudge_tendencies, general_nudge_output, general_nudge_output)
    elif request.param == "all_overlap":
        sources = (nudge_tendencies, nudge_tendencies, nudge_tendencies)

    return sources


def test_MergeNudged(nudge_tendencies, general_nudge_output):

    merged = MergeNudged(nudge_tendencies, LongRunMapper(general_nudge_output))

    assert len(merged) == NTIMES

    item = merged[list(merged.keys())[0]]
    source_vars = chain(
        nudge_tendencies.data_vars.keys(), general_nudge_output.data_vars.keys(),
    )
    for var in source_vars:
        assert var in item


def test_MergeNudged__check_dvar_overlap_fail(overlap_check_fail_datasets):
    with pytest.raises(ValueError):
        MergeNudged._check_dvar_overlap(*overlap_check_fail_datasets)


def test_MergeNudged__rename_vars(nudge_tendencies, general_nudge_output):
    renamed = MergeNudged._rename_vars(
        (nudge_tendencies, general_nudge_output),
        {"air_temperature_tendency_due_to_nudging": "dQ1"},
    )
    assert len(renamed) == 2
    assert "dQ1" in renamed[0]


def test_NudgedStateCheckpoints(nudged_checkpoints):

    mapper = NudgedStateCheckpoints(nudged_checkpoints)

    ds_len = sum([ds.sizes[TIME_NAME] for ds in nudged_checkpoints.values()])
    assert len(mapper) == ds_len

    single_item = nudged_checkpoints["after_physics"].isel(time=0)
    time_key = pd.to_datetime(single_item.time.values).strftime(TIME_FMT)
    single_item = single_item.drop_vars("time")
    item_key = ("after_physics", time_key)
    xr.testing.assert_equal(mapper[item_key], single_item)


class MockMergeNudgedMapper:
    def __init__(self, *nudged_sources):
        self.ds = xr.merge(nudged_sources, join="inner")

    def __getitem__(self, key: str) -> xr.Dataset:
        return self.ds.sel({"time": key})

    def keys(self):
        return self.ds["time"].values


@pytest.fixture
def nudged_source():
    example_da = value_to_xr_darray(10.0)
    air_temperature = value_to_xr_darray(270.0)
    specific_humidity = value_to_xr_darray(0.01)

    net_term_vars = [
        "total_sky_downward_longwave_flux_at_surface",
        "total_sky_downward_shortwave_flux_at_surface",
        "total_sky_upward_longwave_flux_at_surface",
        "total_sky_upward_longwave_flux_at_top_of_atmosphere",
        "total_sky_upward_shortwave_flux_at_surface",
        "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
        "sensible_heat_flux",
        "surface_precipitation_rate",
        "latent_heat_flux",
    ]

    ds_vars = {
        "air_temperature": air_temperature,
        "specific_humidity": specific_humidity,
    }
    ds_vars.update({net_term_var: example_da for net_term_var in net_term_vars})

    return xr.Dataset(ds_vars)


@pytest.fixture
def merged_source(nudged_source):
    additional_vars = {
        "dQ1": value_to_xr_darray(1.0),
        "dQ2": value_to_xr_darray(0.1),
    }
    return nudged_source.assign(additional_vars)


@pytest.fixture
def nudged_mapper(nudged_source):
    return MockMergeNudgedMapper(nudged_source)


@pytest.fixture
def merged_mapper(merged_source):
    return MockMergeNudgedMapper(merged_source)


class MockCheckpointMapper:
    def __init__(self, ds_map):

        self.sources = {key: MockMergeNudgedMapper(ds) for key, ds in ds_map.items()}

    def __getitem__(self, key):
        return self.sources[key[0]][key[1]]


@pytest.fixture
def nudged_checkpoint_mapper_param(request, nudged_source):
    source_map = {request.param[0]: nudged_source, request.param[1]: nudged_source}
    return MockCheckpointMapper(source_map)


@pytest.fixture
def constant_checkpoint_mapper(nudged_source):
    map_sources = {"after_dynamics": nudged_source, "after_physics": nudged_source}
    return MockCheckpointMapper(map_sources)


@pytest.mark.parametrize(
    [
        "nudged_checkpoint_mapper_param",
        "tendency_variables",
        "difference_checkpoints",
        "valid",
        "output_vars",
    ],
    [
        pytest.param(
            ("after_dynamics", "after_physics"),
            None,
            ("after_dynamics", "after_physics"),
            True,
            ["pQ1", "pQ2"],
            id="base",
        ),
        pytest.param(
            ("before_dynamics", "after_physics"),
            None,
            ("after_dynamics", "after_physics"),
            False,
            ["pQ1", "pQ2"],
            id="wrong sources",
        ),
        pytest.param(
            ("after_dynamics", "after_physics"),
            {"Q1": "air_temperature", "Q2": "specific_humidity"},
            ("after_dynamics", "after_physics"),
            True,
            ["Q1", "Q2"],
            id="different term names",
        ),
        pytest.param(
            ("after_dynamics", "after_physics"),
            {"pQ1": "air_temperature", "pQ2": "sphum"},
            ("after_dynamics", "after_physics"),
            False,
            ["pQ1", "pQ2"],
            id="wrong variable name",
        ),
    ],
    indirect=["nudged_checkpoint_mapper_param"],
)
def test_init_nudged_tendencies(
    nudged_checkpoint_mapper_param,
    tendency_variables,
    difference_checkpoints,
    valid,
    output_vars,
    nudged_mapper,
):
    if valid:
        nudged_tendencies_mapper = NudgedFullTendencies(
            nudged_mapper,
            nudged_checkpoint_mapper_param,
            difference_checkpoints,
            tendency_variables,
        )
        safe.get_variables(nudged_tendencies_mapper["20200501.000000"], output_vars)
    else:
        with pytest.raises(KeyError):
            nudged_tendencies_mapper = NudgedFullTendencies(
                nudged_mapper,
                nudged_checkpoint_mapper_param,
                difference_checkpoints,
                tendency_variables,
            )
            safe.get_variables(nudged_tendencies_mapper["20200501.000000"], output_vars)


@pytest.fixture
def nudged_checkpoint_mapper(request):
    source_map = {"after_dynamics": request.param[0], "after_physics": request.param[1]}
    return MockCheckpointMapper(source_map)


@pytest.fixture
def checkpoints():
    return ("after_dynamics", "after_physics")


def value_to_xr_darray(value):
    return xr.DataArray(
        np.full((4, 1), value),
        {
            "time": xr.DataArray(
                [f"2020050{i}.000000" for i in range(1, 5)], dims=["time"]
            ),
            "x": xr.DataArray([0], dims=["x"]),
        },
        ["time", "x"],
    )


nudged_source_1 = xr.Dataset(
    {
        "air_temperature": value_to_xr_darray(270.0),
        "specific_humidity": value_to_xr_darray(0.01),
    }
)
nudged_source_2 = xr.Dataset(
    {
        "air_temperature": value_to_xr_darray(272.0),
        "specific_humidity": value_to_xr_darray(0.005),
    }
)
nudged_source_3 = xr.Dataset(
    {
        "air_temperature": value_to_xr_darray(272.0),
        "specific_humidity": value_to_xr_darray(np.nan),
    }
)


@pytest.fixture
def expected_tendencies(request):
    return {
        "pQ1": xr.DataArray(
            [[request.param[0]]],
            {
                "time": xr.DataArray(["20200501.000000"], dims=["time"]),
                "x": xr.DataArray([0], dims=["x"]),
            },
            ["time", "x"],
        ),
        "pQ2": xr.DataArray(
            [[request.param[1]]],
            {
                "time": xr.DataArray(["20200501.000000"], dims=["time"]),
                "x": xr.DataArray([0], dims=["x"]),
            },
            ["time", "x"],
        ),
    }


@pytest.mark.parametrize(
    ["nudged_checkpoint_mapper", "timestep", "expected_tendencies"],
    [
        pytest.param(
            (nudged_source_1, nudged_source_1), 900, (0.0, 0.0), id="zero tendencies"
        ),
        pytest.param(
            (nudged_source_1, nudged_source_2),
            900.0,
            (2.0 / 900.0, -0.005 / 900.0),
            id="non-zero tendencies",
        ),
        pytest.param(
            (nudged_source_1, nudged_source_2),
            100.0,
            (2.0 / 100.0, -0.005 / 100.0),
            id="different timestep",
        ),
        pytest.param(
            (nudged_source_1, nudged_source_3),
            100.0,
            (2.0 / 100.0, np.nan),
            id="nan data",
        ),
    ],
    indirect=["nudged_checkpoint_mapper", "expected_tendencies"],
)
def test__physics_tendencies(
    nudged_checkpoint_mapper, timestep, expected_tendencies, nudged_mapper, checkpoints
):

    nudged_tendencies_mapper = NudgedFullTendencies(
        nudged_mapper, nudged_checkpoint_mapper
    )

    time = "20200501.000000"

    tendency_variables = {
        "pQ1": "air_temperature",
        "pQ2": "specific_humidity",
    }

    physics_tendencies = nudged_tendencies_mapper._physics_tendencies(
        time, tendency_variables, nudged_checkpoint_mapper, checkpoints, timestep,
    )

    for term in tendency_variables:
        xr.testing.assert_allclose(
            physics_tendencies[term], expected_tendencies[term].sel(time=time)
        )


def test_NudgeToObsState(merged_mapper):
    nudging_variables = {"air_temperature": "dQ1", "specific_humidity": "dQ2"}
    dt = 10
    adjusted_mapper = NudgeToObsState(merged_mapper, nudging_variables, dt)

    assert len(adjusted_mapper) == 4
    key = list(adjusted_mapper.keys())[0]
    output_ds = adjusted_mapper[key]
    source_ds = merged_mapper[key]
    for var in source_ds:
        assert var in output_ds
    xr.testing.assert_allclose(
        adjusted_mapper[key]["air_temperature"],
        merged_mapper[key]["air_temperature"] - dt * merged_mapper[key]["dQ1"],
    )


def test_NudgeToObsFullTendencies(merged_mapper, constant_checkpoint_mapper):
    nudging_variables = {"air_temperature": "dQ1", "specific_humidity": "dQ2"}
    dt = 10
    difference_checkpoints = ("after_dynamics", "after_physics")
    physics_tendency_variables = {"pQ1": "air_temperature", "pQ2": "specific_humidity"}
    full_tendencies = NudgeToObsFullTendencies(
        merged_mapper,
        constant_checkpoint_mapper,
        difference_checkpoints,
        physics_tendency_variables,
        nudging_variables,
        dt,
    )

    assert len(full_tendencies) == 4
    key = list(full_tendencies.keys())[0]
    output_ds = full_tendencies[key]
    source_ds = merged_mapper[key]
    for var in source_ds:
        assert var in output_ds
    # since checkpoint states are equal, pQ = -dQ
    xr.testing.assert_allclose(
        full_tendencies[key]["pQ1"], -merged_mapper[key]["dQ1"],
    )


@pytest.mark.regression
def test_open_merged_nudged(nudged_data_dir):

    merge_files = ("after_dynamics.zarr", "nudging_tendencies.zarr")
    mapper = open_merged_nudged(
        nudged_data_dir, merge_files=merge_files, i_start=4, n_times=6,
    )

    assert len(mapper) == 6


@pytest.mark.regression
def test_open_merged_nudge_to_obs(nudged_data_dir):

    merge_files = ("after_physics.zarr", "nudging_tendencies.zarr")
    rename_vars = {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
    }
    mapper = open_merged_nudge_to_obs(
        nudged_data_dir,
        merge_files=merge_files,
        i_start=4,
        n_times=6,
        rename_vars=rename_vars,
    )

    key = list(mapper.keys())[0]
    mapper[key]["air_temperature"]
    mapper[key]["specific_humidity"]
    mapper[key]["dQ1"]
    mapper[key]["dQ2"]
    assert len(mapper) == 6


@pytest.mark.regression
def test__open_nudging_checkpoints(nudged_data_dir):

    checkpoint_files = ("before_dynamics.zarr", "after_nudging.zarr")
    mapper = _open_nudging_checkpoints(
        nudged_data_dir, checkpoint_files=checkpoint_files
    )
    assert len(mapper) == NTIMES * len(checkpoint_files)


@pytest.mark.regression
def test_open_merged_nudged_full_tendencies(nudged_data_dir):

    open_merged_nudged_kwargs = {"n_times": 6}
    mapper = open_merged_nudged_full_tendencies(
        nudged_data_dir, open_merged_nudged_kwargs=open_merged_nudged_kwargs,
    )

    assert len(mapper) == 6


@pytest.mark.regression
def test_open_merged_nudge_to_obs_full_tendencies(nudged_data_dir):

    open_kwargs = {
        "n_times": 6,
        "rename_vars": {
            "air_temperature_tendency_due_to_nudging": "dQ1",
            "specific_humidity_tendency_due_to_nudging": "dQ2",
        },
    }

    mapper = open_merged_nudge_to_obs_full_tendencies(
        nudged_data_dir, open_merged_nudge_to_obs_kwargs=open_kwargs,
    )

    key = list(mapper.keys())[0]
    mapper[key]["pQ1"]
    mapper[key]["pQ2"]
    mapper[key]["dQ1"]
    mapper[key]["dQ2"]
    assert len(mapper) == 6


@pytest.fixture
def checkpoint_mapping(general_nudge_output):

    item = general_nudge_output.isel(time=slice(0, 1))
    checkpoints = {
        ("before_dynamics", "20160801.001500"): item,
        ("after_dynamics", "20160801.001500"): item,
        ("after_dynamics", "20160801.003000"): item,
    }

    return checkpoints


def test_GroupByTime(checkpoint_mapping):

    test_groupby = GroupByTime(checkpoint_mapping)
    assert len(test_groupby) == 2
    assert list(test_groupby.keys()) == ["20160801.001500", "20160801.003000"]


@pytest.mark.parametrize(
    "time_key, expected_size", [("20160801.001500", 2), ("20160801.003000", 1)]
)
def test_GroupByTime_items(time_key, expected_size, checkpoint_mapping):

    test_groupby = GroupByTime(checkpoint_mapping)

    item = test_groupby[time_key]
    assert "checkpoint" in item.dims
    assert item.sizes["checkpoint"] == expected_size


def test_SubsetTime(nudged_tstep_mapper):

    #     ntimestep_skip_hr = 1  # equivalent to 4 timesteps
    i_start = 4
    n_times = 6
    times = sorted(list(nudged_tstep_mapper.keys()))[4:10]

    subset = SubsetTimes(i_start, n_times, nudged_tstep_mapper)

    assert len(subset) == n_times
    assert times == sorted(list(subset.keys()))


def test_SubsetTime_out_of_order_times(nudged_tstep_mapper):

    times = sorted(list(nudged_tstep_mapper.keys()))[:5]
    shuffled_idxs = [4, 0, 2, 3, 1]
    shuffled_map = {times[i]: nudged_tstep_mapper[times[i]] for i in shuffled_idxs}
    subset = SubsetTimes(0, 2, shuffled_map)

    for i, key in enumerate(sorted(list(subset.keys()))):
        assert key == times[i]
        xr.testing.assert_equal(nudged_tstep_mapper[key], subset[key])


def test_SubsetTime_fail_on_non_subset_key(nudged_tstep_mapper):

    out_of_bounds = sorted(list(nudged_tstep_mapper.keys()))[4]
    subset = SubsetTimes(0, 4, nudged_tstep_mapper)

    with pytest.raises(KeyError):
        subset[out_of_bounds]
