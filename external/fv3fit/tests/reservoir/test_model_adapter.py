import numpy as np
import pytest
import xarray as xr
import fv3fit
from fv3fit.reservoir.transformers.transformer import DoNothingAutoencoder
from fv3fit.reservoir.domain2 import RankXYDivider
from fv3fit.reservoir.adapters import (
    ReservoirDatasetAdapter,
    HybridReservoirDatasetAdapter,
    _transpose_xy_dims,
    split_multi_subdomain_model,
    generate_subdomain_models_for_tile,
    generate_subdomain_models_from_all_tiles,
)

from helpers import get_reservoir_computing_model


@pytest.mark.parametrize(
    "original_dims, reordered_dims",
    [
        (["time", "x", "y", "z"], ["time", "x", "y", "z"]),
        (["time", "y", "x", "z"], ["time", "x", "y", "z"]),
    ],
)
def test__transpose_xy_dims(original_dims, reordered_dims):
    da = xr.DataArray(np.random.rand(5, 7, 7, 8), dims=original_dims)
    assert list(_transpose_xy_dims(da, ("x", "y")).dims) == reordered_dims


def get_8x8_overlapped_model(hybrid: bool):

    divider = RankXYDivider((2, 2), 2, overlap_rank_extent=(8, 8), z_feature_size=6)
    transformer = DoNothingAutoencoder([3, 3])
    return get_reservoir_computing_model(
        divider=divider, encoder=transformer, state_size=25, hybrid=hybrid
    )


def get_single_rank_xarray_data():
    rng = np.random.RandomState(0)
    a = rng.randn(4, 4, 3)  # two variables concatenated to form size 6 latent space
    b = rng.randn(4, 4, 3)

    return xr.Dataset(
        {
            "a": xr.DataArray(a, dims=["x", "y", "z"]),
            "b": xr.DataArray(b, dims=["x", "y", "z"]),
        }
    )


def get_single_rank_xarray_data_with_overlap():
    rng = np.random.RandomState(0)
    a = rng.randn(8, 8, 3)  # two variables concatenated to form size 6 latent space
    b = rng.randn(8, 8, 3)

    return xr.Dataset(
        {
            "a": xr.DataArray(a, dims=["x", "y", "z"]),
            "b": xr.DataArray(b, dims=["x", "y", "z"]),
        }
    )


def test_adapter_predict(regtest):
    hybrid_predictor = get_8x8_overlapped_model(hybrid=True)
    data = get_single_rank_xarray_data_with_overlap()

    model = HybridReservoirDatasetAdapter(
        model=hybrid_predictor,
        input_variables=hybrid_predictor.input_variables,
        output_variables=hybrid_predictor.output_variables,
    )
    nhalo = model.model.rank_divider.overlap
    data_without_overlap = data.isel(
        {"x": slice(nhalo, -nhalo), "y": slice(nhalo, -nhalo)}
    )
    result = model.predict(data_without_overlap)
    print(result, file=regtest)


def test_adapter_increment_state():
    hybrid_predictor = get_8x8_overlapped_model(hybrid=True)
    data = get_single_rank_xarray_data_with_overlap()

    model = HybridReservoirDatasetAdapter(
        model=hybrid_predictor,
        input_variables=hybrid_predictor.input_variables,
        output_variables=hybrid_predictor.output_variables,
    )
    model.reset_state()
    model.increment_state(data)


def test_nonhybrid_adapter_predict(regtest):
    predictor = get_8x8_overlapped_model(hybrid=False)
    data = get_single_rank_xarray_data()

    model = ReservoirDatasetAdapter(
        model=predictor,
        input_variables=predictor.input_variables,
        output_variables=predictor.output_variables,
    )
    nhalo = model.model.rank_divider.overlap
    data_without_overlap = data.isel(
        {"x": slice(nhalo, -nhalo), "y": slice(nhalo, -nhalo)}
    )
    result = model.predict(data_without_overlap)
    print(result, file=regtest)


def test_adapter_dump_and_load(tmpdir):
    predictor = get_8x8_overlapped_model(hybrid=False)
    data = get_single_rank_xarray_data()

    model = ReservoirDatasetAdapter(
        model=predictor,
        input_variables=predictor.input_variables,
        output_variables=predictor.output_variables,
    )
    nhalo = model.model.rank_divider.overlap
    data_without_overlap = data.isel(
        {"x": slice(nhalo, -nhalo), "y": slice(nhalo, -nhalo)}
    )
    model.reset_state()
    result0 = model.predict(data_without_overlap)

    model.dump(str(tmpdir))
    loaded_model = fv3fit.load(str(tmpdir))
    loaded_model.reset_state()
    print(model)
    print(loaded_model)
    result1 = loaded_model.predict(data_without_overlap)
    for r0, r1 in zip(result0, result1):
        np.testing.assert_array_equal(r0, r1)


def test_split_multi_subdomain_model():

    divider = RankXYDivider((2, 2), 2, overlap_rank_extent=(8, 8), z_feature_size=3)
    encoder = DoNothingAutoencoder([3])
    model = get_reservoir_computing_model(
        divider=divider, encoder=encoder, state_size=25, hybrid=True
    )
    data = get_single_rank_xarray_data_with_overlap()["a"]
    hybrid_data = get_single_rank_xarray_data()["a"]

    model.increment_state([data])
    result = model.predict([hybrid_data])

    no_overlap_divider = divider.get_no_overlap_rank_divider()

    # 4 subdomains w/ 6 x 6 x 1 features
    all_subdomain_data = divider.get_all_subdomains(data)
    all_subdomain_hybrid_data = no_overlap_divider.get_all_subdomains(hybrid_data)
    split_models = split_multi_subdomain_model(model)

    # check that each subdomain model matches the expected subdomain
    # decomposed result from the full rank model
    for i, subdomain_model in enumerate(split_models):
        subdomain_data = all_subdomain_data[i]
        subdomain_hybrid_data = all_subdomain_hybrid_data[i]
        subdomain_model.reset_state()
        subdomain_model.increment_state([subdomain_data])
        subdomain_result = subdomain_model.predict([subdomain_hybrid_data])[0]
        expected = no_overlap_divider.get_subdomain(result[0], i)
        np.testing.assert_array_equal(subdomain_result, expected)


@pytest.mark.parametrize(
    "is_hybrid, use_adapter", [(True, True), (False, True), (False, False)]
)
def test_generate_subdomain_models_for_tile(is_hybrid, use_adapter):
    model = get_8x8_overlapped_model(hybrid=is_hybrid)
    adapter_class = (
        HybridReservoirDatasetAdapter if is_hybrid else ReservoirDatasetAdapter
    )
    if use_adapter:
        model = adapter_class(
            model=model,
            input_variables=model.input_variables,
            output_variables=model.output_variables,
        )

    split_models = split_multi_subdomain_model(model)
    assert len(split_models) == 4


# test saved model for single tile
def test_generate_subdomain_models_for_saved_single_tile(tmpdir):
    model = get_8x8_overlapped_model(hybrid=True)
    save_path = str(tmpdir.join("model0"))
    model.dump(save_path)
    generate_subdomain_models_for_tile(save_path, str(tmpdir.join("new_models")))
    for i in range(4):
        fv3fit.load(str(tmpdir.join("new_models").join(f"subdomain_{i}")))


# test saved model for multiple tiles


def test_generate_subdomain_models_for_saved_all_tiles(tmpdir):
    model = get_8x8_overlapped_model(hybrid=True)
    model_map = {}
    for i in range(6):
        save_path = str(tmpdir.join(f"model{i}"))
        model.dump(save_path)
        model_map[i] = save_path

    generate_subdomain_models_from_all_tiles(model_map, str(tmpdir.join("new_models")))
    for i in range(24):
        fv3fit.load(str(tmpdir.join("new_models").join(f"subdomain_{i}")))
