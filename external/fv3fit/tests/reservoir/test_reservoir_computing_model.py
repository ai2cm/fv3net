from fv3fit.reservoir.domain2 import RankXYDivider
from fv3fit.reservoir.transformers import DoNothingAutoencoder, TransformerGroup
import numpy as np
import pytest

from scipy import sparse

from fv3fit.reservoir import (
    ReservoirComputingModel,
    Reservoir,
    ReservoirHyperparameters,
)
from .convenience import get_ReservoirComputingModel


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
    predictor = get_ReservoirComputingModel()
    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)

    assert loaded_predictor.rank_divider is not None


def test_dump_load_preserves_matrices(tmpdir):
    predictor = get_ReservoirComputingModel()
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
    predictor = get_ReservoirComputingModel(
        state_size=state_size,
        rank_divider=rank_divider,
        autoencoder=transformer,
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

    predictor = get_ReservoirComputingModel()
    predictor.reset_state()

    n_times = 20

    ts_sync = [
        np.ones((n_times, *predictor.rank_divider.rank_extent, 1))
        for v in predictor.input_variables
    ]
    predictor.synchronize(ts_sync)
    for i in range(10):
        prediction0 = predictor.predict()

    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)
    loaded_predictor.reset_state()

    loaded_predictor.synchronize(ts_sync)
    for i in range(10):
        prediction1 = loaded_predictor.predict()
    np.testing.assert_array_almost_equal(prediction0[0], prediction1[0])


@pytest.mark.skip(reason="HybridReservoirComputingModel for different variables broken")
def test_HybridReservoirComputingModel_dump_load(tmpdir):
    state_size = 1000
    rank_divider = RankXYDivider((2, 2), (2, 2))

    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=rank_divider.flat_subdomain_len)
    assert reservoir
    # TODO: If removing the large block diagonal form of the readout,
    # this needs to be updated


#     n_total_inputs = (
#         state_size + rank_divider.subdomain_xy_size_without_overlap ** 2
#     ) * rank_divider.n_subdomains
#     readout = ReservoirComputingReadout(
#         coefficients=np.random.rand(n_total_inputs, input_size),
#         intercepts=np.random.rand(input_size),
#     )
#     hybrid_predictor = HybridReservoirComputingModel(
#         input_variables=["a", "b"],
#         hybrid_variables=["c"],
#         output_variables=["a", "b"],
#         reservoir=reservoir,
#         readout=readout,
#         rank_divider=rank_divider,
#         autoencoder=DoNothingAutoencoder([1]),
#     )
#     hybrid_predictor.reset_state()
#     ts_sync = [
#         np.ones((input_size, hybrid_predictor.rank_divider.n_subdomains,))
#         for i in range(20)
#     ]

#     hybrid_predictor.synchronize(ts_sync)

#     # Training data always has a feature dim, even if it's size 1
#     hybrid_input = [
#         np.random.rand(*rank_divider.rank_extent, 1),
#     ]
#     prediction0 = hybrid_predictor.predict(hybrid_input)

#     output_path = f"{str(tmpdir)}/predictor"
#     hybrid_predictor.dump(output_path)
#     loaded_hybrid_predictor = HybridReservoirComputingModel.load(output_path)
#     loaded_hybrid_predictor.reset_state()

#     loaded_hybrid_predictor.synchronize(ts_sync)
#     prediction1 = loaded_hybrid_predictor.predict(hybrid_input)

#     np.testing.assert_array_almost_equal(prediction0, prediction1)


# def test_HybridReservoirComputingModel_concat_readout_inputs():
#     # TODO: Complex test, will be simplified in future when the model is refactored
#     input_size = 4
#     state_size = 3
#     hyperparameters = ReservoirHyperparameters(
#         state_size=state_size,
#         adjacency_matrix_sparsity=0.9,
#         spectral_radius=1.0,
#         input_coupling_sparsity=0,
#     )
#     rank_divider = default_rank_divider

#     reservoir = Reservoir(
#         hyperparameters, input_size=rank_divider.subdomain_size_with_overlap
#     )
#     n_hybrid_inputs = (
#        rank_divider.subdomain_xy_size_without_overlap ** 2 * rank_divider.n_subdomains
#     )
#     readout = ReservoirComputingReadout(
#         coefficients=np.random.rand(state_size + n_hybrid_inputs, input_size),
#         intercepts=np.random.rand(input_size),
#     )
#     hybrid_predictor = HybridReservoirComputingModel(
#         input_variables=["a", "b"],
#         hybrid_variables=["c"],
#         output_variables=["a", "b"],
#         reservoir=reservoir,
#         readout=readout,
#         rank_divider=rank_divider,
#         autoencoder=DoNothingAutoencoder([1]),
#     )
#     hybrid_predictor.reset_state()

#     # hidden state of each subdomain is constant array of its index
#     hybrid_predictor.reservoir_model.reservoir.state = np.array(
#         [np.arange(rank_divider.n_subdomains) for zfeature in range(state_size)]
#     )

#     # partitioner indexing goes (0,0) -> 0, (1,0)-> 1, etc.
#     hybrid_inputs = np.array([[0, -2], [-1, -3]])
#     flat_hybrid_inputs = hybrid_predictor.rank_divider.flatten_subdomains_to_columns(
#         hybrid_inputs, with_overlap=False
#     )
#     flattened_readout_input = hybrid_predictor._concatenate_readout_inputs(
#         hybrid_predictor.reservoir_model.reservoir.state, flat_hybrid_inputs
#     )
#     np.testing.assert_array_equal(
#         flattened_readout_input,
#         np.array([0, 0, 0, 0, 1, 1, 1, -1, 2, 2, 2, -2, 3, 3, 3, -3]),
#     )
