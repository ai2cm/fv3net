import fsspec
import xarray as xr
from sklearn.externals import joblib
from sklearn.utils import parallel_backend

import state_io


def open_sklearn_model(url):
    # Load the model
    with fsspec.open(url, "rb") as f:
        return joblib.load(f)


def predict(model, state):
    stacked = state.stack(sample=["x", "y"])
    with parallel_backend("threading", n_jobs=1):
        output = model.predict(stacked, "sample").unstack("sample")
    return output


def update(model, state, dt):
    renamed = state_io.rename_to_restart(state)
    state = xr.Dataset(renamed)

    tend = predict(model, state)

    updated = state.assign(
        sphum=state["sphum"] + tend.dQ2 * dt, T=state.T + tend.dQ1 * dt
    )

    return state_io.rename_to_orig(updated), state_io.rename_to_orig(tend)


if __name__ == "__main__":
    import sys

    state_path = sys.argv[1]
    model = open_sklearn_model(sys.argv[2])

    with open(state_path, "rb") as f:
        data = state_io.load(f)

    tile = data[0]
    preds = update(model, tile, dt=1)
    print(preds)
