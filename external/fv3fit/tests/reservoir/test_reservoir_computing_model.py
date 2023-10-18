from fv3fit.reservoir.domain2 import RankXYDivider
from fv3fit.reservoir.transformers import DoNothingAutoencoder, TransformerGroup
import numpy as np
import pytest

from scipy import sparse

from fv3fit.reservoir import (
    ReservoirComputingModel,
    Reservoir,
    ReservoirHyperparameters,
    HybridReservoirComputingModel,
)

from helpers import get_reservoir_computing_model


class MultiOutputMeanRegressor:
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs

    def predict(self, input):
        # returns array of shape (1, n_outputs), with each element
        # the mean of the input vector elements
        return np.full(self.n_outputs, np.mean(input)).reshape(1, -1)


def _sparse_allclose(A, B, atol=1e-8):
    # https://stackoverflow.com/a/47771340
    r1, c1, v1 = sparse.find(A)
    r2, c2, v2 = sparse.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)
    if index_match == 0:
        return False
    else:
        return np.allclose(v1, v2, atol=atol)


def test_dump_load_optional_attrs(tmpdir):
    predictor = get_reservoir_computing_model()
    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)

    assert loaded_predictor.rank_divider is not None


def test_dump_load_preserves_matrices(tmpdir):
    predictor = get_reservoir_computing_model()
    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)

    loaded_predictor = ReservoirComputingModel.load(output_path)
    assert _sparse_allclose(loaded_predictor.reservoir.W_in, predictor.reservoir.W_in,)
    assert _sparse_allclose(
        loaded_predictor.reservoir.W_res, predictor.reservoir.W_res,
    )
    np.testing.assert_array_almost_equal(
        loaded_predictor.readout.coefficients, predictor.readout.coefficients,
    )
    np.testing.assert_array_almost_equal(
        loaded_predictor.readout.intercepts, predictor.readout.intercepts
    )


@pytest.mark.parametrize("nz, nvars", [(1, 1), (3, 1), (3, 3), (1, 3)])
def test_prediction_shape(nz, nvars):
    transformer = DoNothingAutoencoder([nz for var in range(nvars)])
    rank_divider = RankXYDivider(
        (2, 2), 0, rank_extent=(2, 2), z_feature_size=transformer.n_latent_dims
    )
    input_size = rank_divider.flat_subdomain_len
    state_size = 1000
    transformer.encode([np.ones((input_size, nz)) for v in range(nvars)])
    variables = [f"var{v}" for v in range(nvars)]
    predictor = get_reservoir_computing_model(
        state_size=state_size,
        divider=rank_divider,
        encoder=transformer,
        variables=variables,
    )
    predictor.reset_state()

    for v in range(nvars):
        assert predictor.predict()[v].shape == (
            *predictor.rank_divider.rank_extent,
            nz,
        )


def test_ReservoirComputingModel_state_increment():
    rank_divider = RankXYDivider((1, 1), 0, rank_extent=(2, 2), z_feature_size=1)
    input_size = rank_divider.flat_subdomain_len
    state_size = 3
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    reservoir.W_in = sparse.csc_matrix(np.ones(reservoir.W_in.shape))
    reservoir.W_res = sparse.csc_matrix(np.ones(reservoir.W_res.shape))

    readout = MultiOutputMeanRegressor(n_outputs=input_size)

    input = [(0.25 * np.ones((*rank_divider.rank_extent, 1)))]
    transformer = DoNothingAutoencoder([1])
    transformers = TransformerGroup(
        input=transformer, output=transformer, hybrid=transformer
    )
    transformers.input.encode(input)
    predictor = ReservoirComputingModel(
        input_variables=["a", "b"],
        output_variables=["a", "b"],
        reservoir=reservoir,
        readout=readout,
        rank_divider=rank_divider,
        transformers=transformers,
    )

    predictor.reset_state()
    predictor.increment_state(input)
    state_before_prediction = predictor.reservoir.state
    encoded_prediction = predictor.transformers.input.encode(predictor.predict())
    predictor.increment_state(input)

    np.testing.assert_array_almost_equal(
        encoded_prediction, np.tanh(np.ones(input[0].shape))
    )
    assert not np.allclose(state_before_prediction, predictor.reservoir.state)


def test_prediction_after_load(tmpdir):

    predictor = get_reservoir_computing_model()
    predictor.reset_state()

    n_times = 20

    ts_sync = [
        np.ones((n_times, *predictor.rank_divider.rank_extent, 1))
        for v in predictor.input_variables
    ]
    predictor.synchronize(ts_sync)
    prediction0 = predictor.predict()

    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)
    loaded_predictor.reset_state()

    loaded_predictor.synchronize(ts_sync)
    prediction1 = loaded_predictor.predict()
    np.testing.assert_array_almost_equal(prediction0[0], prediction1[0])


def test_state_preserved_after_load(tmpdir):

    predictor = get_reservoir_computing_model()
    predictor.reset_state()

    n_times = 20

    rng = np.random.RandomState(0)
    ts_sync = [
        rng.randn(n_times, *predictor.rank_divider.rank_extent, 1)
        for v in predictor.input_variables
    ]
    predictor.synchronize(ts_sync)

    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    predictor.reservoir.dump_state(f"{output_path}/reservoir")
    loaded_predictor = ReservoirComputingModel.load(output_path)

    np.testing.assert_array_equal(
        predictor.reservoir.state, loaded_predictor.reservoir.state
    )


def test_HybridReservoirComputingModel_dump_load(tmpdir):
    model = get_reservoir_computing_model(hybrid=True)
    model.reset_state()

    n_times = 20

    ts_sync = [
        np.ones((n_times, *model.rank_divider.rank_extent, 1))
        for v in model.input_variables
    ]
    model.synchronize(ts_sync)
    prediction0 = model.predict([ts[-1] for ts in ts_sync])

    output_path = f"{str(tmpdir)}/predictor"
    model.dump(output_path)
    loaded_predictor = HybridReservoirComputingModel.load(output_path)
    loaded_predictor.reset_state()

    loaded_predictor.synchronize(ts_sync)
    prediction1 = loaded_predictor.predict([ts[-1] for ts in ts_sync])
    np.testing.assert_array_almost_equal(prediction0[0], prediction1[0])
    np.testing.assert_array_equal(
        model._hybrid_input_mask, loaded_predictor._hybrid_input_mask
    )


def test_HybridReservoirComputingModel_hybrid_mask():
    model = get_reservoir_computing_model(hybrid=True)

    model.reset_state()
    model._hybrid_input_mask = np.zeros_like(model._hybrid_input_mask)

    n_times = 20

    ts_sync = [
        np.random.randn(n_times, *model.rank_divider.rank_extent, 1)
        for v in model.input_variables
    ]
    model.synchronize(ts_sync)
    prediction0 = model.predict([ts[-1] for ts in ts_sync])

    model.reset_state()
    model.synchronize(ts_sync)
    ts_sync = [
        np.random.randn(n_times, *model.rank_divider.rank_extent, 1)
        for v in model.input_variables
    ]
    prediction1 = model.predict([ts[-1] for ts in ts_sync])
    np.testing.assert_array_almost_equal(prediction0[0], prediction1[0])
