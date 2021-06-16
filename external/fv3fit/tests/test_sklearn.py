from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib


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
