import fsspec
from sklearn.externals import joblib
from sklearn.utils import parallel_backend
import xarray as xr


__all__ = ["open_model", "predict", "update"]

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def open_model(url):
    # Load the model
    with fsspec.open(url, "rb") as f:
        return joblib.load(f)


def predict(model, state):
    stacked = state.stack(sample=["x", "y"])
    with parallel_backend("threading", n_jobs=1):
        output = model.predict(stacked, "sample").unstack("sample")
    logger.info(output)
    return output


def update(model, state, dt):
    state = xr.Dataset(state)
    tend = predict(model, state)
    updated = state.assign(
        specific_humidity=state["specific_humidity"] + tend["dQ2"] * dt,
        air_temperature=state["air_temperature"] + tend["dQ1"] * dt,
    )
    return {key: updated[key] for key in updated}, {key: tend[key] for key in tend}
