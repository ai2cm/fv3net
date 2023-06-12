from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import fv3fit.train
import xarray as xr
import vcm.testing


def test_numpy_version(regtest):
    print(np.version.full_version, file=regtest)


def test_random_state_uniform_reproducibility():
    seed = 10
    fv3fit.set_random_seed(seed)
    random = np.random.RandomState(0)
    n_samples = 500
    X0 = random.uniform(size=(n_samples, 5))

    fv3fit.set_random_seed(seed)
    random = np.random.RandomState(0)
    n_samples = 500
    X1 = random.uniform(size=(n_samples, 5))
    assert np.array_equal(X0, X1)


def test_random_state_second_output_reproducibility(regtest):
    random = np.random.RandomState(0)
    n_samples = 500
    _ = random.uniform(size=(n_samples, 5))
    X = random.uniform(size=(n_samples, 5))
    print(joblib.hash(X), file=regtest)


def test_data_generation_reproducibility(regtest):
    """
    This sequence of operations occurs in another test, and at one point
    seemed not to be reproducible between systems.
    """
    fv3fit.set_random_seed(1)
    random = np.random.RandomState(0)
    n_sample, n_feature = int(5e3), 2

    def sample_func():
        return xr.DataArray(
            random.uniform(size=(n_sample, n_feature)), dims=["sample", "feature_dim"]
        )

    _ = sample_func()
    data_array = sample_func()
    test_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    for result in vcm.testing.checksum_dataarray_mapping(test_dataset):
        print(result, file=regtest)


def set_random_state_train_and_predict(seed):
    fv3fit.set_random_seed(seed)
    regressor = RandomForestRegressor(random_state=0, n_jobs=None)
    random = np.random.RandomState(0)
    n_samples = 500
    X = random.uniform(size=(n_samples, 5))
    y = random.uniform(size=(n_samples, 2))
    with joblib.parallel_backend("loky", n_jobs=8):
        regressor.fit(X, y)
    X_test = random.uniform(size=(n_samples, 5))
    y_test = regressor.predict(X_test)
    return y_test


def test_random_forest_reproducibility():
    seed = 10
    prediction_0 = set_random_state_train_and_predict(seed)
    prediction_1 = set_random_state_train_and_predict(seed)
    assert np.array_equal(prediction_0, prediction_1)


def test_random_forest_n_jobs_can_exceed_n_estimators():
    regressor = RandomForestRegressor(random_state=0, n_estimators=1, n_jobs=None)
    random = np.random.RandomState(0)
    n_samples = 100
    X = random.uniform(size=(n_samples, 5))
    y = random.uniform(size=(n_samples, 2))
    with joblib.parallel_backend("loky", n_jobs=8):
        regressor.fit(X, y)
    X_test = random.uniform(size=(n_samples, 5))
    regressor.predict(X_test)
