import numpy as np
from scipy import sparse

from sklearn.dummy import DummyRegressor

from fv3fit.reservoir.predictor import ReservoirComputingModel, _square_even_terms
from fv3fit.reservoir.reservoir import Reservoir, ReservoirHyperparameters


def test__square_even_terms():
    x = np.arange(4)
    np.testing.assert_array_equal(_square_even_terms(x), np.array([0, 1, 4, 3]))


class MultiOutputMeanRegressor:
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs

    def predict(self, input):
        # returns vector of size n_outputs, with each element
        # the mean of the input vector elements
        return np.full(self.n_outputs, np.mean(input))


def _sparse_allclose(A, B, atol=1e-8):
    # https://stackoverflow.com/a/47771340
    r1, c1, v1 = sparse.find(A)
    r2, c2, v2 = sparse.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)
    if index_match == 0:
        return False
    else:
        return np.allclose(v1, v2, atol=atol)


def test_dump_load_preserves_reservoir(tmpdir):
    hyperparameters = ReservoirHyperparameters(
        input_size=10,
        state_size=150,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    predictor = ReservoirComputingModel(
        reservoir=reservoir, readout=DummyRegressor(strategy="constant", constant=-1.0),
    )
    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)

    loaded_predictor = ReservoirComputingModel.load(output_path)
    assert _sparse_allclose(loaded_predictor.reservoir.W_in, predictor.reservoir.W_in)
    assert _sparse_allclose(loaded_predictor.reservoir.W_res, predictor.reservoir.W_res)


def test_ReservoirComputingModel_state_increment():
    input_size = 2
    hyperparameters = ReservoirHyperparameters(
        input_size=2,
        state_size=3,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    reservoir.W_in = sparse.coo_matrix(np.ones(reservoir.W_in.shape))
    reservoir.W_res = sparse.coo_matrix(np.ones(reservoir.W_res.shape))

    readout = MultiOutputMeanRegressor(n_outputs=input_size)
    predictor = ReservoirComputingModel(
        reservoir=reservoir, readout=readout, square_half_hidden_state=False
    )

    predictor.reservoir.reset_state()
    predictor.reservoir.increment_state(np.array([0.5, 0.5]))

    prediction = predictor.predict()
    np.testing.assert_array_almost_equal(prediction, np.tanh(np.array([1.0, 1.0])))


def test_prediction_after_load(tmpdir):
    input_size = 15
    hyperparameters = ReservoirHyperparameters(
        input_size=input_size,
        state_size=1000,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    readout = MultiOutputMeanRegressor(n_outputs=input_size)
    predictor = ReservoirComputingModel(
        reservoir=reservoir, readout=readout, square_half_hidden_state=False
    )
    predictor.reservoir.reset_state()
    for i in range(10):
        predictor.reservoir.increment_state(np.ones(input_size))
    prediction0 = predictor.predict()

    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)
    loaded_predictor.reservoir.reset_state()
    for i in range(10):
        loaded_predictor.reservoir.increment_state(np.ones(input_size))
    prediction1 = loaded_predictor.predict()

    np.testing.assert_array_almost_equal(prediction0, prediction1)
