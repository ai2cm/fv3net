from typing import Any, Mapping, Sequence, Union

from loaders import mappers
from vcm import safe
from .._shared import ModelTrainingConfig, pack
from .._shared import transforms

STACK_DIMS = ["x", "y", "tile"]
SAMPLE_DIM = "sample"


def get_target_transformer(data_path: str, config: ModelTrainingConfig):
    transform_creation_func = globals()[config.target_transform or "standard_scaler"]
    return transform_creation_func(data_path, config)


def mass_scaler(data_path: str, config: ModelTrainingConfig):
    normalization_sample_vars = ["pressure_thickness_of_atmospheric_layer"]


def standard_scaler(data_path, config):
    norm_sample_vars = config.output_variables
    norm_data_sample = _load_normalization_data(
        data_path,
        norm_sample_vars,
        config.batch_kwargs["mapping_function"],
        config.batch_kwargs["mapping_kwargs"],
    )
    y, _ = pack(norm_data_sample, SAMPLE_DIM)
    transformer = transforms.StandardScaler()
    transformer.fit(y)
    return transformer


def _load_normalization_data(
    data_path: str,
    vars: Sequence[str],
    mapping_func_name: str,
    mapping_kwargs: Mapping[str, Any],
    sample_selection: Mapping[str, Union[slice, int]] = None,
    stack_dims: Sequence[str] = None,
):
    mapping_func = getattr(mappers, mapping_func_name)
    mapper = mapping_func(data_path, **mapping_kwargs)
    # sample_selection = sample_selection or {"x": 0, "y": 0, "tile": 0}
    sample_key = sorted(list(mapper.keys()))[0]
    sample = safe.get_variables(mapper[sample_key], vars).load()
    stacked_sample = safe.stack_once(
        sample,
        dim=SAMPLE_DIM,
        dims=stack_dims or STACK_DIMS,
        allowed_broadcast_dims=["z"],
    ).transpose()
    return stacked_sample
