import numpy as np
import pytest
from scipy import sparse
from fv3fit.reservoir import Reservoir, ReservoirHyperparameters


def test_matrices_and_state_correct_dims():
    N_input, N_res = 10, 100
    hyperparameters = ReservoirHyperparameters(
        input_size=N_input,
        state_size=N_res,
        adjacency_matrix_sparsity=0.05,
        spectral_radius=0.2,
    )
    reservoir = Reservoir(hyperparameters)
    assert reservoir.W_in.shape == (N_res, N_input)
    assert (reservoir.W_in * np.ones(N_input)).shape == (N_res,)
    assert reservoir.W_res.shape == (N_res, N_res)
    reservoir.increment_state(np.ones(N_input))
    assert reservoir.state.shape == (N_res,)


@pytest.mark.parametrize("adjacency_matrix_sparsity", [0.01, 0.1])
def test_Wres_sparsity(adjacency_matrix_sparsity):
    N_input, N_res = 10, 100
    hyperparameters = ReservoirHyperparameters(
        input_size=N_input,
        state_size=N_res,
        adjacency_matrix_sparsity=adjacency_matrix_sparsity,
        spectral_radius=0.2,
    )
    reservoir = Reservoir(hyperparameters)
    assert (
        reservoir.W_res.count_nonzero() == (1 - adjacency_matrix_sparsity) * N_res ** 2
    )


def test_spectral_radius():
    radius = 0.6
    hyperparameters = ReservoirHyperparameters(
        input_size=10,
        state_size=100,
        adjacency_matrix_sparsity=0.8,
        spectral_radius=radius,
    )
    reservoir = Reservoir(hyperparameters)
    np.testing.assert_almost_equal(
        sparse.linalg.eigs(
            reservoir.W_res, return_eigenvectors=False, k=1, which="LM"
        ).item(),
        radius,
    )


def test_Win_equal_connections_per_input():
    input_coupling_sparsity = 0.2
    hyperparameters = ReservoirHyperparameters(
        input_size=100,
        state_size=1000,
        adjacency_matrix_sparsity=0.8,
        spectral_radius=0.6,
        input_coupling_sparsity=input_coupling_sparsity,
    )
    reservoir = Reservoir(hyperparameters)
    nonzero_per_col = [
        reservoir.W_in.getcol(i).count_nonzero()
        for i in range(hyperparameters.input_size)
    ]
    assert np.unique(nonzero_per_col).item() == 800
    assert len(np.unique(nonzero_per_col)) == 1


def test_increment_state():
    hyperparameters = ReservoirHyperparameters(
        input_size=2,
        state_size=3,
        adjacency_matrix_sparsity=0.0,
        input_coupling_sparsity=0,
        spectral_radius=1.0,
    )
    reservoir = Reservoir(hyperparameters)

    reservoir.W_in = sparse.coo_matrix(np.ones(reservoir.W_in.shape))
    reservoir.W_res = sparse.identity(hyperparameters.state_size)

    reservoir.reset_state()
    input = np.array([0.5, 0.5])

    reservoir.increment_state(input)
    np.testing.assert_array_almost_equal(
        reservoir.state, np.tanh(np.array([1.0, 1.0, 1.0]))
    )

    reservoir.increment_state(input)
    elem = np.tanh(1.0) + 1.0
    np.testing.assert_array_almost_equal(
        reservoir.state, np.tanh(np.array([elem, elem, elem]))
    )
