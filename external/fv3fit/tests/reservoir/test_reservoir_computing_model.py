import numpy as np
from scipy import sparse

from sklearn.dummy import DummyRegressor

from fv3fit.reservoir import (
    ReservoirComputingModel,
    ReservoirComputingReadout,
    Reservoir,
    ReservoirHyperparameters,
    HybridReservoirComputingModel,
)


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


def test_dump_load_preserves_reservoir(tmpdir):
    hyperparameters = ReservoirHyperparameters(
        input_size=10,
        state_size=150,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    readout = ReservoirComputingReadout(
        linear_regressor=DummyRegressor(strategy="constant", constant=-1.0),
        square_half_hidden_state=False,
    )
    predictor = ReservoirComputingModel(reservoir=reservoir, readout=readout,)
    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)

    loaded_predictor = ReservoirComputingModel.load(output_path)
    assert _sparse_allclose(loaded_predictor.reservoir.W_in, predictor.reservoir.W_in)
    assert _sparse_allclose(loaded_predictor.reservoir.W_res, predictor.reservoir.W_res)


def test_prediction_shape():
    input_size = 15
    hyperparameters = ReservoirHyperparameters(
        input_size=input_size,
        state_size=1000,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    lr = DummyRegressor(strategy="constant", constant=np.ones(input_size))
    lr.fit(reservoir.state.reshape(1, -1), np.ones((1, input_size)))
    readout = ReservoirComputingReadout(
        linear_regressor=lr, square_half_hidden_state=True,
    )
    predictor = ReservoirComputingModel(reservoir=reservoir, readout=readout,)
    # ReservoirComputingModel.predict reshapes the prediction to remove
    # the first dim of length 1 (sklearn regressors predict 2D arrays)
    assert predictor.predict().shape == (input_size,)


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

    readout = ReservoirComputingReadout(
        linear_regressor=MultiOutputMeanRegressor(n_outputs=input_size),
        square_half_hidden_state=False,
    )
    predictor = ReservoirComputingModel(reservoir=reservoir, readout=readout,)

    predictor.reservoir.reset_state()
    predictor.reservoir.increment_state(np.array([0.5, 0.5]))
    state_before_prediction = predictor.reservoir.state
    prediction = predictor.predict()
    np.testing.assert_array_almost_equal(prediction, np.tanh(np.array([1.0, 1.0])))
    assert not np.allclose(state_before_prediction, predictor.reservoir.state)


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
    readout = ReservoirComputingReadout(
        linear_regressor=MultiOutputMeanRegressor(n_outputs=input_size),
        square_half_hidden_state=True,
    )
    predictor = ReservoirComputingModel(reservoir=reservoir, readout=readout,)
    predictor.reservoir.reset_state()

    ts_sync = [np.ones(input_size) for i in range(20)]
    predictor.reservoir.synchronize(ts_sync)
    for i in range(10):
        prediction0 = predictor.predict()

    output_path = f"{str(tmpdir)}/predictor"
    predictor.dump(output_path)
    loaded_predictor = ReservoirComputingModel.load(output_path)
    loaded_predictor.reservoir.reset_state()

    loaded_predictor.reservoir.synchronize(ts_sync)
    for i in range(10):
        prediction1 = loaded_predictor.predict()

    np.testing.assert_array_almost_equal(prediction0, prediction1)


class MockImperfectModel:
    def __init__(self, offset: float):
        self.offset = offset

    def predict(self, input):
        return input + self.offset


def test_hybrid_prediction_after_load(tmpdir):
    imperfect_model = MockImperfectModel(offset=0.1)
    input_size = 15
    hyperparameters = ReservoirHyperparameters(
        input_size=input_size,
        state_size=1000,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    readout = ReservoirComputingReadout(
        linear_regressor=MultiOutputMeanRegressor(n_outputs=input_size),
        square_half_hidden_state=True,
    )
    hybrid_predictor = HybridReservoirComputingModel(
        reservoir=reservoir, readout=readout, imperfect_model=imperfect_model
    )

    ts_sync = [np.ones(input_size) for i in range(20)]
    hybrid_predictor.reservoir.synchronize(ts_sync)
    predictions_0 = [
        ts_sync[-1],
    ]
    for i in range(10):
        predictions_0.append(hybrid_predictor.predict(predictions_0[-1]))

    output_path = f"{str(tmpdir)}/hybrid_predictor"
    hybrid_predictor.dump(output_path)
    loaded_hybrid_predictor = HybridReservoirComputingModel.load(output_path)

    loaded_hybrid_predictor.reservoir.synchronize(ts_sync)
    predictions_1 = [
        ts_sync[-1],
    ]
    for i in range(10):
        predictions_1.append(loaded_hybrid_predictor.predict(predictions_1[-1]))
    np.testing.assert_array_almost_equal(
        np.array(predictions_0), np.array(predictions_1)
    )


def test_hybrid_gives_different_results():
    imperfect_model = MockImperfectModel(offset=0.1)
    input_size = 15
    hyperparameters = ReservoirHyperparameters(
        input_size=input_size,
        state_size=1000,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    readout = ReservoirComputingReadout(
        linear_regressor=MultiOutputMeanRegressor(n_outputs=input_size),
        square_half_hidden_state=True,
    )
    predictor = ReservoirComputingModel(reservoir=reservoir, readout=readout)
    hybrid_predictor = HybridReservoirComputingModel(
        reservoir=reservoir, readout=readout, imperfect_model=imperfect_model
    )

    ts_sync = [np.ones(input_size) for i in range(20)]
    predictor.reservoir.synchronize(ts_sync)
    hybrid_predictor.reservoir.synchronize(ts_sync)

    predictions, predictions_hybrid = [ts_sync[-1]], [ts_sync[-1]]

    for i in range(10):
        predictions.append(predictor.predict())
        predictions_hybrid.append(hybrid_predictor.predict(predictions_hybrid[-1]))

    predictions = np.array(predictions)
    predictions_hybrid = np.array(predictions_hybrid)
    assert predictions.shape == predictions_hybrid.shape
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, predictions, predictions_hybrid
    )
