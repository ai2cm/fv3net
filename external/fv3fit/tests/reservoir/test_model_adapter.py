import numpy as np
import pytest
import xarray as xr
import fv3fit
from fv3fit.reservoir.transformers.transformer import DoNothingAutoencoder
from fv3fit.reservoir.domain2 import RankXYDivider
from fv3fit.reservoir.readout import ReservoirComputingReadout
from fv3fit.reservoir import (
    Reservoir,
    ReservoirHyperparameters,
    HybridReservoirComputingModel,
    ReservoirComputingModel,
)
from fv3fit.reservoir.adapters import (
    ReservoirDatasetAdapter,
    HybridReservoirDatasetAdapter,
    _transpose_xy_dims,
)


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


def get_initialized_model(hybrid: bool):
    # expects rank size (including halos) in latent space
    divider = RankXYDivider((2, 2), 2, overlap_rank_extent=(8, 8), z_feature_size=6)
    autoencoder = DoNothingAutoencoder([3, 3])

    state_size = 25
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=1,
    )
    reservoir = Reservoir(hyperparameters, input_size=divider.flat_subdomain_len)

    no_overlap_divider = divider.get_no_overlap_rank_divider()
    # multiplied by the number of subdomains since it's a combined readout

    if hybrid:
        coefs_feature_size = state_size + no_overlap_divider.flat_subdomain_len
    else:
        coefs_feature_size = state_size
    readout = ReservoirComputingReadout(
        coefficients=np.random.rand(
            divider.n_subdomains,
            coefs_feature_size,
            no_overlap_divider.flat_subdomain_len,
        ),
        intercepts=np.random.rand(
            divider.n_subdomains, no_overlap_divider.flat_subdomain_len
        ),
    )
    if hybrid:
        predictor = HybridReservoirComputingModel(
            input_variables=["a", "b"],
            output_variables=["a", "b"],
            hybrid_variables=["a", "b"],
            reservoir=reservoir,
            readout=readout,
            rank_divider=divider,
            autoencoder=autoencoder,
        )
    else:
        predictor = ReservoirComputingModel(
            input_variables=["a", "b"],
            output_variables=["a", "b"],
            reservoir=reservoir,
            readout=readout,
            rank_divider=divider,
            autoencoder=autoencoder,
        )
    predictor.reset_state()

    return predictor


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
    hybrid_predictor = get_initialized_model(hybrid=True)
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
    hybrid_predictor = get_initialized_model(hybrid=True)
    data = get_single_rank_xarray_data_with_overlap()

    model = HybridReservoirDatasetAdapter(
        model=hybrid_predictor,
        input_variables=hybrid_predictor.input_variables,
        output_variables=hybrid_predictor.output_variables,
    )
    model.reset_state()
    model.increment_state(data)


def test_nonhybrid_adapter_predict(regtest):
    predictor = get_initialized_model(hybrid=False)
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
    predictor = get_initialized_model(hybrid=False)
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
