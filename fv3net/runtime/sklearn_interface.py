import fsspec
from sklearn.externals import joblib
from sklearn.utils import parallel_backend


__all__ = ["open_model", "predict", "update"]


def open_model(url):
    # Load the model
    with fsspec.open(url, "rb") as f:
        return joblib.load(f)


def predict(model, state):
    stacked = state.stack(sample=["x", "y"])
    with parallel_backend("threading", n_jobs=1):
        output = model.predict(stacked, "sample").unstack("sample")
    return output


def update(model, state, dt):
    tend = predict(model, state)
    updated = state.assign(
        specific_humidity=state["specific_humidity"] + tend.dQ2 * dt,
        air_temperature=state["air_temperature"] + tend.dQ1 * dt,
    )
    return {key: state[key] for key in updated}, {key: state[key] for key in tend}
