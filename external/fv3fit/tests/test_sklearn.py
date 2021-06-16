from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import fv3fit.train
import xarray as xr
import vcm.testing


def test_random_state_uniform_reproducibility(regtest):
    random = np.random.RandomState(0)
    n_samples = 500
    X = random.randn(n_samples, 5)
    print(joblib.hash(X), file=regtest)


def test_random_state_second_output_reproducibility(regtest):
    random = np.random.RandomState(0)
    n_samples = 500
    _ = random.randn(n_samples, 5)
    X = random.randn(n_samples, 5)
    print(joblib.hash(X), file=regtest)


def test_data_generation_reproducibility(regtest):
    """
    This sequence of operations occurs in another test, and at one point
    seemed not to be reproducible between systems.
    """
    fv3fit.train.set_random_seed(1)
    random = np.random.RandomState(0)
    n_sample, n_feature = int(5e3), 2

    def sample_func():
        return xr.DataArray(
            random.randn(n_sample, n_feature), dims=["sample", "feature_dim"]
        )

    _ = sample_func()
    data_array = sample_func()
    test_dataset = xr.Dataset(data_vars={"var_in": data_array, "var_out": data_array})
    for result in vcm.testing.checksum_dataarray_mapping(test_dataset):
        print(result, file=regtest)


def test_random_forest_reproducibility(regtest):
    regressor = RandomForestRegressor(random_state=0, n_jobs=None)
    random = np.random.RandomState(0)
    n_samples = 500
    X = random.randn(n_samples, 5)
    y = random.randn(n_samples, 2)
    with joblib.parallel_backend("loky", n_jobs=8):
        regressor.fit(X, y)
    X_test = random.randn(n_samples, 5)
    y_test = regressor.predict(X_test)
    print(joblib.hash(y_test), file=regtest)


def test_random_forest_n_jobs_can_exceed_n_estimators():
    regressor = RandomForestRegressor(random_state=0, n_estimators=1, n_jobs=None)
    random = np.random.RandomState(0)
    n_samples = 100
    X = random.randn(n_samples, 5)
    y = random.randn(n_samples, 2)
    with joblib.parallel_backend("loky", n_jobs=8):
        regressor.fit(X, y)
    X_test = random.randn(n_samples, 5)
    regressor.predict(X_test)
