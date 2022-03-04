from fv3fit._shared.config import SliceConfig
from fv3fit._shared.packer import clip_sample
import tensorflow as tf
from typing import Hashable, Mapping, Dict, Callable, Optional, Sequence, Tuple
from toolz.functoolz import curry
import loaders.typing


@curry
def apply_to_mapping(
    tensor_func: Callable[[tf.Tensor], tf.Tensor], data: Mapping[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    return {name: tensor_func(tensor) for name, tensor in data.items()}


@curry
def ensure_nd(n: int, tensor: tf.Tensor) -> tf.Tensor:
    """
    Given a tensor that may be n-dimensional or (n-1)-dimensional, return
    a tensor that is n-dimensional, adding a length 1 dimension to the end if needed.
    """
    if len(tensor.shape) == n - 1:
        return tensor[..., None]
    elif len(tensor.shape) == n:
        return tensor
    else:
        raise ValueError(
            f"received a tensor with {len(tensor.shape)} dimensions, "
            f"cannot make it {n}-dimensional"
        )


@curry
def select_keys(
    variable_names: Sequence[str], data: Mapping[str, tf.Tensor]
) -> Tuple[tf.Tensor, ...]:
    return tuple(data[name] for name in variable_names)


def get_Xy_dataset(
    input_variables: Sequence[str],
    output_variables: Sequence[str],
    clip_config: Optional[Mapping[Hashable, SliceConfig]],
    n_dims: int,
    data: tf.data.Dataset,
):
    """
    Given a tf.data.Dataset with mappings from variable name to samples,
    return a tf.data.Dataset whose entries are two tuples, the first containing the
    requested input variables and the second containing
    the requested output variables.
    """
    data = data.map(apply_to_mapping(ensure_nd(n_dims)))
    if clip_config is not None:
        y_source = data.map(clip_sample(clip_config))
    else:
        y_source = data
    y = y_source.map(select_keys(output_variables))
    X = data.map(select_keys(input_variables))
    return tf.data.Dataset.zip((X, y))


def tfdataset_from_batches(batches: loaders.typing.Batches) -> tf.data.Dataset:
    def gen():
        for ds in batches:
            yield {key: tf.convert_to_tensor(val) for key, val in ds.items()}

    try:
        sample = next(iter(gen()))
    except StopIteration:
        sample = {}
    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            key: tf.TensorSpec(val.shape, dtype=val.dtype)
            for key, val in sample.items()
        },
    ).unbatch()
