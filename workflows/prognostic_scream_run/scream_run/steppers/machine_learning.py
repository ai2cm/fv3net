import dataclasses
from typing import (
    Hashable,
    Iterable,
    MutableMapping,
    Mapping,
    Optional,
    Sequence,
    Set,
    cast,
    Tuple,
)
import fv3fit
import xarray as xr
import vcm

State = MutableMapping[Hashable, xr.DataArray]
SPHUM = "qv"


@dataclasses.dataclass
class MachineLearningConfig:
    """Machine learning configurations

    Attributes:
        models: list of URLs to fv3fit models.
        diagnostic_ml: do not apply ML tendencies if true.
        scaling: if given, scale the outputs by the given factor. This is a manually
            defined alteration of the model, and should not be used for
            normalization.
        mse_conserving_limiter (optional): whether to use MSE-conserving humidity
                limiter. Defaults to True.
    Example::

        MachineLearningConfig(
            models=["gs://vcm-ml-scratch/test-annak/ml-pipeline-output"],
            diagnostic_ml=False,
        )

    """

    models: Sequence[str] = dataclasses.field(default_factory=list)
    diagnostic_ml: bool = False
    scaling: Mapping[str, float] = dataclasses.field(default_factory=dict)
    mse_conserving_limiter: bool = True


class MultiModelAdapter:
    def __init__(
        self,
        models: Iterable[fv3fit.Predictor],
        scaling: Optional[Mapping[str, float]] = None,
        mse_conserving_limiter: bool = True,
    ):
        """
        Args:
            models: models for which to combine predictions
            scaling: if given, scale the predictions by the given factor
            mse_conserving_limiter (optional): whether to use MSE-conserving humidity
                limiter. Defaults to True.
        """
        self.models = models
        if scaling is None:
            self._scaling: Mapping[str, float] = {}
        else:
            self._scaling = scaling
        self.mse_conserving_limiter = mse_conserving_limiter

    @property
    def input_variables(self) -> Set[str]:
        all_inputs = []  # type: ignore
        for model in self.models:
            all_inputs.extend(model.input_variables)
        return set(all_inputs)

    def predict(self, arg: xr.Dataset) -> xr.Dataset:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(arg))
        ds = xr.merge(predictions)
        for var, scale in self._scaling.items():
            ds[var] *= scale
        return ds


def open_model(config: MachineLearningConfig) -> MultiModelAdapter:
    model_paths = config.models
    models = []
    for path in model_paths:
        model = cast(fv3fit.Predictor, fv3fit.load(path))
        models.append(model)
    return MultiModelAdapter(
        models,
        scaling=config.scaling,
        mse_conserving_limiter=config.mse_conserving_limiter,
    )


def predict(model: MultiModelAdapter, state: State, dt: float) -> State:
    """Given ML model and state, return prediction"""
    state_loaded = {key: state[key] for key in model.input_variables}
    ds = xr.Dataset(state_loaded)  # type: ignore
    output = model.predict(ds)
    output = enforce_non_negative_humidity(
        output, state, dt, model.mse_conserving_limiter
    )
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}


def non_negative_sphum(
    sphum: xr.DataArray, dQ1: xr.DataArray, dQ2: xr.DataArray, dt: float
) -> Tuple[xr.DataArray, xr.DataArray]:
    delta = dQ2 * dt
    reduction_ratio = (-sphum) / (dt * dQ2)  # type: ignore
    dQ1_updated = xr.where(sphum + delta >= 0, dQ1, reduction_ratio * dQ1)
    dQ2_updated = xr.where(sphum + delta >= 0, dQ2, reduction_ratio * dQ2)
    return dQ1_updated, dQ2_updated


def update_moisture_tendency_to_ensure_non_negative_humidity(
    sphum: xr.DataArray, q2: xr.DataArray, dt: float
) -> xr.DataArray:
    return xr.where(sphum + q2 * dt >= 0, q2, -sphum / dt)


def update_temperature_tendency_to_conserve_mse(
    q1: xr.DataArray, q2_old: xr.DataArray, q2_new: xr.DataArray
) -> xr.DataArray:
    mse_tendency = vcm.moist_static_energy_tendency(q1, q2_old)
    q1_new = vcm.temperature_tendency(mse_tendency, q2_new)
    return q1_new


def non_negative_sphum_mse_conserving(
    sphum: xr.DataArray, q2: xr.DataArray, dt: float, q1: Optional[xr.DataArray] = None
) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    q2_new = update_moisture_tendency_to_ensure_non_negative_humidity(sphum, q2, dt)
    if q1 is not None:
        q1_new = update_temperature_tendency_to_conserve_mse(q1, q2, q2_new)
    else:
        q1_new = None
    return q2_new, q1_new


def enforce_non_negative_humidity(
    prediction: dict, state: State, dt: float, mse_conserving_limiter: bool = True,
):
    dQ1_initial = prediction.get("dQ1", xr.zeros_like(state[SPHUM]))
    dQ2_initial = prediction.get("dQ2", xr.zeros_like(state[SPHUM]))
    if mse_conserving_limiter:
        dQ2_updated, dQ1_updated = non_negative_sphum_mse_conserving(
            state[SPHUM], dQ2_initial, dt, q1=dQ1_initial,
        )
    else:
        dQ1_updated, dQ2_updated = non_negative_sphum(
            state[SPHUM], dQ1_initial, dQ2_initial, dt,
        )
    if "dQ1" in prediction:
        prediction.update({"dQ1": dQ1_updated})
    if "dQ2" in prediction:
        prediction.update({"dQ2": dQ2_updated})
    return prediction
