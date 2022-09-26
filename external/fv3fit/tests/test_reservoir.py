import numpy as np
import pytest
from scipy import sparse
from fv3fit.reservoir.reservoir import Reservoir, ReservoirHyperparameters


def test_matrices_and_state_correct_dims():
    N_input, N_res = 10, 1000
    hyperparameters = ReservoirHyperparameters(
        input_dim=N_input,
        reservoir_state_dim=N_res,
        sparsity=0.05,
        square_half=True,
        spectral_radius=0.2,
    )
    reservoir = Reservoir(hyperparameters)
    assert reservoir.W_in.shape == (N_res, N_input)
    assert reservoir.W_res.shape == (N_res, N_res)
    reservoir.increment_state(np.ones(N_input))
    assert reservoir.state.shape == (N_res,)


@pytest.mark.parametrize("sparsity", [0.01, 0.1])
def test_Wres_sparsity(sparsity):
    N_input, N_res = 10, 1000
    hyperparameters = ReservoirHyperparameters(
        input_dim=N_input,
        reservoir_state_dim=N_res,
        sparsity=sparsity,
        square_half=True,
        spectral_radius=0.2,
    )
    reservoir = Reservoir(hyperparameters)
    assert reservoir.W_res.count_nonzero() == (1 - sparsity) * N_res ** 2


def test_spectral_radius():
    radius = 0.6
    hyperparameters = ReservoirHyperparameters(
        input_dim=10,
        reservoir_state_dim=1000,
        sparsity=0.8,
        square_half=True,
        spectral_radius=radius,
    )
    reservoir = Reservoir(hyperparameters)
    np.testing.assert_almost_equal(
        max(abs(sparse.linalg.eigs(reservoir.W_res)[0])), radius
    )


def test_Win_equal_connections_per_input():
    input_coupling_sparsity = 0.2
    hyperparameters = ReservoirHyperparameters(
        input_dim=10,
        reservoir_state_dim=1000,
        sparsity=0.8,
        square_half=True,
        spectral_radius=0.6,
        input_coupling_sparsity=input_coupling_sparsity,
    )
    reservoir = Reservoir(hyperparameters)
    nonzero_per_row = [row.count_nonzero() for row in reservoir.W_in]
    assert len(np.unique(nonzero_per_row)) == 1
