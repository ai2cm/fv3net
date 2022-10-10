import numpy as np
from sklearn.dummy import DummyRegressor

from fv3fit.reservoir.predictor import ReservoirPredictor
from fv3fit.reservoir.reservoir import Reservoir, ReservoirHyperparameters


def test_ReservoirPredictor_state_increment():

    hyperparameters = ReservoirHyperparameters(
        input_dim=2,
        reservoir_state_dim=2,
        sparsity=0.0,
        res_scaling=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters)
    reservoir.W_in = np.ones(reservoir.W_in.shape)
    reservoir.W_res = np.ones((reservoir.W_res.shape))

    predictor = ReservoirPredictor(
        reservoir=reservoir, linreg=DummyRegressor(strategy="constant", constant=-1.0)
    )

    reservoir.reset_state()
    reservoir.increment_state(np.array([0.5, 0.5]))

    prediction = predictor.predict()
    np.testing.assert_array_almost_equal(
        prediction, -1.0 * np.tanh(np.array([[0.5, 0.5], [0.5, 0.5]]))
    )
