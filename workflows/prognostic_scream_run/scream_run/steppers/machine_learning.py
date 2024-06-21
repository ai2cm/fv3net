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
)
import xarray as xr
import vcm
import logging

State = MutableMapping[Hashable, xr.DataArray]
SPHUM = "qv"


logger = logging.getLogger(__name__)


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
        models: Iterable,
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
    import fv3fit
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


def _to_state(ds: xr.Dataset) -> State:
    return {key: cast(xr.DataArray, ds[key]) for key in ds.data_vars}


def predict(model: MultiModelAdapter, state: State) -> State:
    """Given ML model and state, return prediction"""
    state_loaded = {key: state[key] for key in model.input_variables}
    ds = xr.Dataset(state_loaded)  # type: ignore
    return _to_state(model.predict(ds))


def predict_with_qv_constraint(
    model: MultiModelAdapter, state: State, dt: float
) -> State:
    """Given ML model and state, return prediction"""
    output = predict(model, state)
    return enforce_non_negative_humidity(
        output, state, dt, model.mse_conserving_limiter
    )


def enforce_non_negative_humidity(
    prediction: State, state: State, dt: float, mse_conserving_limiter: bool = True,
):
    dQ1_initial = prediction.get("dQ1", xr.zeros_like(state[SPHUM]))
    dQ2_initial = prediction.get("dQ2", xr.zeros_like(state[SPHUM]))
    logger.info(f"Non-neg humidity dtype: {dQ1_initial.dtype}")
    if mse_conserving_limiter:
        dQ2_updated, dQ1_updated = vcm.non_negative_sphum_mse_conserving(
            state[SPHUM], dQ2_initial, dt, q1=dQ1_initial,
        )
    else:
        dQ1_updated, dQ2_updated = vcm.non_negative_sphum(
            state[SPHUM], dQ1_initial, dQ2_initial, dt,
        )
    if "dQ1" in prediction:
        prediction.update({"dQ1": dQ1_updated})
    if "dQ2" in prediction:
        prediction.update({"dQ2": dQ2_updated})
    logger.info(f"Non-neg humidity dtype: {dQ1_updated.dtype}")
    return prediction
