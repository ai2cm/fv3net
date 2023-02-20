import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
import unittest.mock
import pytest
import xarray as xr
import joblib
import io

from fv3fit.sklearn._random_forest import pack, SklearnWrapper
from fv3fit._shared.scaler import ManualScaler
from fv3fit._shared import PackerConfig, SliceConfig
from fv3fit.tfdataset import tfdataset_from_batches


def test_flatten():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a})

    ans = pack(ds, [sample_dim])[0]
    assert ans.shape == (nz, 2 * nx * ny)


def test_flatten_1d_input():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a.isel(x=0, y=0)})

    ans = pack(ds, [sample_dim])[0]
    assert ans.shape == (nz, nx * ny + 1)


def test_flatten_same_order():
    nx, ny = 10, 4
    x = xr.DataArray(np.arange(nx * ny).reshape((nx, ny)), dims=["sample", "feature"])

    ds = xr.Dataset({"a": x, "b": x.T})
    sample_dim = "sample"
    a = pack(ds[["a"]], [sample_dim])[0]
    b = pack(ds[["b"]], [sample_dim])[0]

    np.testing.assert_allclose(a, b)


def _get_sklearn_wrapper(scale_factor=None, dumps_returns: bytes = b"HEY!"):
    model = unittest.mock.Mock()
    model.predict.return_value = np.ones(shape=(1,))
    model.dumps.return_value = dumps_returns

    if scale_factor:
        scaler = ManualScaler(np.array([scale_factor]))
    else:
        scaler = None

    wrapper = SklearnWrapper(
        input_variables=["x"], output_variables=["y"], model=model, n_jobs=1
    )
    wrapper.target_scaler = scaler
    return wrapper


def test_SklearnWrapper_fit_predict_scaler(scale=2.0):
    wrapper = _get_sklearn_wrapper(scale)
    dims = ["unstacked_dim", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(tfdataset_from_batches([data]))
    output = wrapper.predict(data)
    assert pytest.approx(1 / scale) == output["y"].item()


def test_fitting_SklearnWrapper_does_not_fit_scaler():
    """SklearnWrapper should use pre-computed scaling factors when fitting data

    In other words, calling the .fit method of wrapper should not call the
    .fit its scaler attribute.
    """

    model = unittest.mock.Mock()
    scaler = unittest.mock.Mock()

    wrapper = SklearnWrapper(
        input_variables=["x"], output_variables=["y"], model=model, n_jobs=1
    )
    wrapper.target_scaler = scaler

    dims = ["sample_", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(tfdataset_from_batches([data]))
    scaler.fit.assert_not_called()


@pytest.mark.parametrize(
    "scale_factor", [2.0, None],
)
def test_SklearnWrapper_serialize_predicts_the_same(tmpdir, scale_factor):

    # Setup wrapper
    if scale_factor:
        scaler = ManualScaler(np.array([scale_factor]))
    else:
        scaler = None
    model = LinearRegression()
    wrapper = SklearnWrapper(
        input_variables=["x"], output_variables=["y"], model=model, n_jobs=1
    )
    wrapper.target_scaler = scaler

    # setup input data
    dims = ["unstacked_dim", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(tfdataset_from_batches([data]))

    # serialize/deserialize
    path = str(tmpdir)
    wrapper.dump(path)

    loaded = wrapper.load(path)
    xr.testing.assert_equal(loaded.predict(data), wrapper.predict(data))


def fit_wrapper_with_columnar_data():
    nz = 2
    model = DummyRegressor(strategy="constant", constant=np.arange(nz))
    wrapper = SklearnWrapper(
        input_variables=["a"], output_variables=["b"], model=model, n_jobs=1
    )

    dims = ["sample", "z"]
    shape = (4, nz)
    arr = np.arange(np.prod(shape)).reshape(shape)
    input_data = xr.Dataset({"a": (dims, arr), "b": (dims, arr + 1)})
    wrapper.fit(tfdataset_from_batches([input_data]))
    return input_data, wrapper


def get_unstacked_data():
    dims = ["x", "y", "z"]
    shape = (2, 2, 2)
    arr = np.arange(np.prod(shape)).reshape(shape)
    input_data = xr.Dataset({"a": (dims, arr), "b": (dims, arr + 1)})
    return input_data


def test_predict_is_deterministic(regtest):
    """Tests that fitting/predicting with a model is deterministic

    If this fails, look for non-deterministic logic (e.g. converting sets to lists)
    """
    input_data, wrapper = fit_wrapper_with_columnar_data()
    output = wrapper.predict(input_data)
    print(joblib.hash(np.asarray(output["b"])), file=regtest)


def test_predict_returns_unstacked_dims():
    # 2D output dims same as input dims
    _, wrapper = fit_wrapper_with_columnar_data()
    input_data = get_unstacked_data()
    prediction = wrapper.predict(input_data)
    assert prediction.dims == input_data.dims


def fit_wrapper_with_gridcell_data():
    model = LinearRegression()
    wrapper = SklearnWrapper(
        input_variables=["a"],
        output_variables=["b"],
        model=model,
        n_jobs=1,
        predict_columns=False,
    )
    dims = [
        "sample",
    ]
    shape = (10,)
    arr = np.arange(shape[0])
    input_data = xr.Dataset({"a": (dims, arr), "b": (dims, arr + 1.0)})
    wrapper.fit(tfdataset_from_batches([input_data]))
    return input_data, wrapper


def test_predict_columns_false():
    _, wrapper = fit_wrapper_with_gridcell_data()
    input_data = get_unstacked_data()
    # works
    _ = wrapper.predict(input_data)
    wrapper.predict_columns = True
    with pytest.raises(ValueError):
        # wrong number of features
        _ = wrapper.predict(input_data)


def test_SklearnWrapper_fit_predict_with_clipped_input_data():
    nz = 5
    model = DummyRegressor(strategy="constant", constant=np.arange(nz))
    wrapper = SklearnWrapper(
        input_variables=["a", "b"],
        output_variables=["c"],
        model=model,
        n_jobs=1,
        packer_config=PackerConfig({"a": SliceConfig(2, None)}),
    )

    dims = ["sample", "z"]
    shape = (4, nz)
    arr = np.arange(np.prod(shape)).reshape(shape)
    input_data = xr.Dataset(
        {"a": (dims, arr), "b": (dims[:-1], arr[:, 0]), "c": (dims, arr + 1)}
    )
    wrapper.fit(tfdataset_from_batches([input_data]))
    wrapper.predict(input_data)


def test_SklearnWrapper_raises_not_implemented_error_with_clipped_output_data():
    nz = 5
    model = DummyRegressor(strategy="constant", constant=np.arange(nz))
    with pytest.raises(NotImplementedError):
        SklearnWrapper(
            input_variables=["a", "b"],
            output_variables=["c"],
            model=model,
            n_jobs=1,
            packer_config=PackerConfig({"c": SliceConfig(2, None)}),
        )


class OldSklearnWrapper(SklearnWrapper):
    def __init__(
        self, input_variables, output_variables, model, n_jobs, n_regressors, **kwargs
    ):
        self._n_regressors = n_regressors
        super().__init__(input_variables, output_variables, model, n_jobs, **kwargs)

    def _dump_regressor(self):
        # how regressor ensemble was saved
        regressors = [self.model for _ in range(self._n_regressors)]
        regressor_components = {"regressors": regressors, "n_jobs": self.n_jobs}
        f = io.BytesIO()
        joblib.dump(regressor_components, f)
        return f.getvalue()


def _get_old_sklearn_wrapper(n_regressors):
    nz = 5
    model = DummyRegressor(strategy="constant", constant=np.arange(nz))
    old_wrapper = OldSklearnWrapper(
        input_variables=["a"],
        output_variables=["b"],
        model=model,
        n_jobs=1,
        n_regressors=n_regressors,
    )
    # setup input data; wrappers must be fit to be dumped
    dims = ["sample", "z"]
    data = xr.Dataset({"a": (dims, np.ones((1, 5))), "b": (dims, np.ones((1, 5)))})
    old_wrapper.fit(tfdataset_from_batches([data]))
    return old_wrapper


def test_SklearnWrapper_loads_backwards_compatible(tmpdir):
    old_wrapper = _get_old_sklearn_wrapper(n_regressors=1)
    path = str(tmpdir)
    old_wrapper.dump(path)
    SklearnWrapper.load(path)


def test_SklearnWrapper_loads_backwards_incompatible_multiple(tmpdir):
    old_wrapper = _get_old_sklearn_wrapper(n_regressors=2)
    path = str(tmpdir)
    old_wrapper.dump(path)
    with pytest.raises(ValueError):
        SklearnWrapper.load(path)
