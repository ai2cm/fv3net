import numpy as np
from scipy import sparse

from fv3fit.reservoir import (
    ReservoirComputingModel,
    ReservoirComputingReadout,
    Reservoir,
    ReservoirHyperparameters,
)
from fv3fit.reservoir.one_dim import (
    HybridReservoirComputingModel,
    ReservoirOnlyDomainPredictor,
    HybridDomainPredictor,
    SubdomainConfig,
)
from fv3fit.reservoir.config import ReadoutHyperparameters


def generic_readout(**readout_kwargs):
    readout_hyperparameters = ReadoutHyperparameters(
        linear_regressor_kwargs={}, square_half_hidden_state=False
    )
    return ReservoirComputingReadout(readout_hyperparameters, **readout_kwargs)


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


class MockImperfectModel:
    def __init__(self, offset: float):
        self.offset = offset

    def predict(self, input):
        return input + self.offset


def test_hybrid_prediction_after_load(tmpdir):
    imperfect_model = MockImperfectModel(offset=0.1)
    input_size = 15
    state_size = 1000
    hyperparameters = ReservoirHyperparameters(
        state_size=state_size,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size,)

    readout = generic_readout()
    readout.fit(np.random.rand(1, state_size + input_size), np.ones((1, input_size)))

    hybrid_predictor = HybridReservoirComputingModel(
        reservoir=reservoir, readout=readout,
    )

    ts_sync = [np.ones(input_size) for i in range(20)]
    hybrid_predictor.reservoir.synchronize(ts_sync)
    predictions_0 = [
        ts_sync[-1],
    ]
    for i in range(10):
        predictions_0.append(
            hybrid_predictor.predict(predictions_0[-1], imperfect_model=imperfect_model)
        )

    output_path = f"{str(tmpdir)}/hybrid_predictor"
    hybrid_predictor.dump(output_path)
    loaded_hybrid_predictor = HybridReservoirComputingModel.load(output_path)

    loaded_hybrid_predictor.reservoir.synchronize(ts_sync)
    predictions_1 = [
        ts_sync[-1],
    ]
    for i in range(10):
        predictions_1.append(
            loaded_hybrid_predictor.predict(
                predictions_1[-1], imperfect_model=imperfect_model
            )
        )
    np.testing.assert_array_almost_equal(
        np.array(predictions_0), np.array(predictions_1)
    )


def test_hybrid_gives_different_results():
    imperfect_model = MockImperfectModel(offset=0.1)
    input_size = 15
    hyperparameters = ReservoirHyperparameters(
        state_size=1000,
        adjacency_matrix_sparsity=0.9,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    reservoir = Reservoir(hyperparameters, input_size=input_size)
    readout = MultiOutputMeanRegressor(n_outputs=input_size)
    predictor = ReservoirComputingModel(reservoir=reservoir, readout=readout)
    hybrid_predictor = HybridReservoirComputingModel(
        reservoir=reservoir, readout=readout,
    )

    ts_sync = [np.ones(input_size) for i in range(20)]
    predictor.reservoir.synchronize(ts_sync)
    hybrid_predictor.reservoir.synchronize(ts_sync)

    predictions, predictions_hybrid = [ts_sync[-1]], [ts_sync[-1]]

    for i in range(10):
        predictions.append(predictor.predict())
        predictions_hybrid.append(
            hybrid_predictor.predict(
                predictions_hybrid[-1], imperfect_model=imperfect_model
            )
        )

    predictions = np.array(predictions)
    predictions_hybrid = np.array(predictions_hybrid)
    assert predictions.shape == predictions_hybrid.shape
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, predictions, predictions_hybrid
    )


def create_domain_predictor(
    type, domain_size, subdomain_size, subdomain_overlap, n_jobs=1
):
    # joblib overhead is very slow and sometimes tests fail in CI when
    # running in parallel with joblib.
    # n_jobs can be manually set to >1 in local testing to check that
    # parallelism is working as intended.
    n_subdomains = domain_size // subdomain_size
    input_size = subdomain_size + 2 * subdomain_overlap
    hyperparameters = ReservoirHyperparameters(
        state_size=20,
        adjacency_matrix_sparsity=0.0,
        spectral_radius=1.0,
        input_coupling_sparsity=0,
    )
    subdomain_predictors = []
    for i in range(n_subdomains):
        reservoir = Reservoir(hyperparameters, input_size=input_size,)
        reservoir.reset_state(input_shape=(input_size,))
        readout = MultiOutputMeanRegressor(n_outputs=subdomain_size)
        if type == "reservoir_only":
            subdomain_predictors.append(
                ReservoirComputingModel(reservoir=reservoir, readout=readout,)
            )

        elif type == "hybrid":
            subdomain_predictors.append(
                HybridReservoirComputingModel(reservoir=reservoir, readout=readout,)
            )

    if type == "reservoir_only":
        return ReservoirOnlyDomainPredictor(
            subdomain_predictors=subdomain_predictors,
            subdomain_config=SubdomainConfig(
                size=subdomain_size, overlap=subdomain_overlap
            ),
            n_jobs=n_jobs,
        )
    elif type == "hybrid":
        return HybridDomainPredictor(
            subdomain_predictors=subdomain_predictors,
            subdomain_config=SubdomainConfig(
                size=subdomain_size, overlap=subdomain_overlap
            ),
            n_jobs=n_jobs,
        )
    else:
        raise ValueError("domain predictor must be of type 'reservoir_only' or 'hybrid")


def test_DomainPredictor_synchronize_updates_states():
    domain_size = 6
    subdomain_size = 3
    subdomain_overlap = 1
    domain_predictor = create_domain_predictor(
        "reservoir_only", domain_size, subdomain_size, subdomain_overlap
    )
    burnin = np.arange(domain_size * 10).reshape(10, domain_size)

    states_init = domain_predictor.states
    domain_predictor.synchronize(data=burnin)
    states_after_synch = domain_predictor.states
    for state_init, state_after_synch in zip(states_init, states_after_synch):
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, state_init, state_after_synch
        )


def test_ReservoirOnlyDomainPredictor_updates_state():
    domain_size = 6
    subdomain_size = 3
    subdomain_overlap = 1

    domain_predictor = create_domain_predictor(
        "reservoir_only", domain_size, subdomain_size, subdomain_overlap, n_jobs=1
    )
    initial_inputs = np.arange(domain_size * 10).reshape(10, domain_size)
    domain_predictor.synchronize(data=initial_inputs)
    states_init = domain_predictor.states
    domain_predictor.predict()
    states_after_predict = domain_predictor.states
    for state_init, state_after_predict in zip(states_init, states_after_predict):
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            state_init,
            state_after_predict,
        )


def test_HybridDomainPredictor_updates_state():
    domain_size = 6
    subdomain_size = 3
    subdomain_overlap = 1
    imperfect_model = MockImperfectModel(offset=0.1)

    domain_predictor = create_domain_predictor(
        "hybrid", domain_size, subdomain_size, subdomain_overlap
    )
    initial_inputs = np.arange(domain_size * 10).reshape(10, domain_size)
    domain_predictor.synchronize(data=initial_inputs[:-1])
    states_init = domain_predictor.states
    domain_predictor.predict(
        input_state=initial_inputs[-1], imperfect_model=imperfect_model
    )
    states_after_predict = domain_predictor.states
    for state_init, state_after_predict in zip(states_init, states_after_predict):
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            state_init,
            state_after_predict,
        )


def test_HybridDomainPredictor_predict_utilizes_imperfect_prediction():
    domain_size = 6
    subdomain_size = 2
    subdomain_overlap = 1
    hybrid_domain_predictor = create_domain_predictor(
        "hybrid", domain_size, subdomain_size, subdomain_overlap
    )

    imperfect_model_0 = MockImperfectModel(offset=0.1)
    imperfect_model_1 = MockImperfectModel(offset=1)

    initial_inputs = np.arange(domain_size * 10).reshape(10, domain_size)

    hybrid_domain_predictor.synchronize(data=initial_inputs[:-1])
    prediction_0 = hybrid_domain_predictor.predict(
        input_state=initial_inputs[-1], imperfect_model=imperfect_model_0
    )

    hybrid_domain_predictor.synchronize(data=initial_inputs[:-1])
    prediction_1 = hybrid_domain_predictor.predict(
        input_state=initial_inputs[-1], imperfect_model=imperfect_model_1
    )

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, prediction_0, prediction_1
    )
