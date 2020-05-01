import fsspec
from sklearn.externals import joblib
from sklearn.utils import parallel_backend
import xarray as xr


__all__ = ["open_model", "predict", "update"]


def open_model(url):
    # Load the model
    with fsspec.open(url, "rb") as f:
        return joblib.load(f)


def predict(model, state):
    """Given ML model and state, make tendency prediction.

    Args:
        model (fv3net.regression.sklearn.wrapper.SklearnWrapper): trained ML model
        state (xr.Dataset): atmospheric state

    Returns:
        (xr.Dataset): ML model prediction
    """
    stacked = state.stack(sample=["x", "y"])
    with parallel_backend("threading", n_jobs=1):
        output = model.predict(stacked, "sample").unstack("sample")
    return output


def update(model, state, dt):
    """Update state with ML model prediction of tendencies.

    Args:
        model (fv3net.regression.sklearn.wrapper.SklearnWrapper): trained ML model
        state (xr.Dataset): atmospheric state
        dt (float): timestep (seconds)

    Returns:
        (xr.Dataset, xr.Dataset): tuple of updated state and predicted tendencies
    """
    tend = predict(model, state)
    with xr.set_options(keep_attrs=True):
        updated = state.assign(
            specific_humidity=state["specific_humidity"] + tend["dQ2"] * dt,
            air_temperature=state["air_temperature"] + tend["dQ1"] * dt,
        )
    return updated, tend
