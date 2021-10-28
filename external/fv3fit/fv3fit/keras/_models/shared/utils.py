from fv3fit._shared.packer import ArrayPacker
from fv3fit._shared.stacking import SAMPLE_DIM_NAME
import tensorflow as tf
from typing import Dict, List, Optional, Sequence, Type
import xarray as xr
from fv3fit.emulation.layers import StandardNormLayer, StandardDenormLayer, NormLayer
import fv3fit._shared


def get_input_vector(
    packer: ArrayPacker, n_window: Optional[int] = None, series: bool = True,
):
    """
    Given a packer, return a list of input layers with one layer
    for each input used by the packer, and a list of output tensors which are
    the result of packing those input layers.

    Args:
        packer
        n_window: required if series is True, number of timesteps in a sample
        series: if True, returned inputs have shape [n_window, n_features], otherwise
            they are 1D [n_features] arrays
    """
    features = [packer.feature_counts[name] for name in packer.pack_names]
    if series:
        if n_window is None:
            raise TypeError("n_window is required if series is True")
        input_layers = [
            tf.keras.layers.Input(shape=[n_window, n_features])
            for n_features in features
        ]
    else:
        input_layers = [
            tf.keras.layers.Input(shape=[n_features]) for n_features in features
        ]
    packed = tf.keras.layers.Concatenate()(input_layers)
    return input_layers, packed


def count_features(
    names, batch: xr.Dataset, sample_dims=(SAMPLE_DIM_NAME,)
) -> Dict[str, int]:
    """
    Returns counts of the number of features for each variable name.

    Args:
        names: dataset keys to be unpacked
        batch: dataset containing representatively-shaped data for the given names,
            last dimension should be the feature dimension.
    """

    _, feature_index = fv3fit._shared.pack(batch[names], sample_dims=sample_dims)
    return fv3fit._shared.count_features(feature_index)


def _fit_norm_layer(
    cls: Type[NormLayer],
    names: Sequence[str],
    layers: Sequence[tf.Tensor],
    batch: xr.Dataset,
    sample_dims: Sequence[str] = (SAMPLE_DIM_NAME,),
) -> Sequence[NormLayer]:
    out: List[NormLayer] = []
    input_features = count_features(names=names, batch=batch, sample_dims=sample_dims)
    for name, layer in zip(names, layers):
        norm = cls(name=f"standard_normalize_{name}")
        selection: List[Optional[slice]] = [
            slice(None, None) for _ in batch[name].shape
        ]
        if input_features[name] == 1:
            selection = selection + [None]
        norm.fit(batch[name].values[tuple(selection)])
        out.append(norm(layer))
    return out


def standard_normalize(
    names: Sequence[str],
    layers: Sequence[tf.Tensor],
    batch: xr.Dataset,
    sample_dims: Sequence[str] = (SAMPLE_DIM_NAME,),
) -> Sequence[tf.Tensor]:
    """
    Apply standard scaling to a series of layers based on mean and standard
    deviation from a batch of data.

    Args:
        names: variable name in batch of each layer in layers
        layers: input tensors to be scaled by scaling layers
        batch: reference data for mean and standard deviation
    
    Returns:
        normalized_layers: standard-scaled tensors
    """
    return _fit_norm_layer(
        StandardNormLayer,
        names=names,
        layers=layers,
        batch=batch,
        sample_dims=sample_dims,
    )


def standard_denormalize(
    names: Sequence[str],
    layers: Sequence[tf.Tensor],
    batch: xr.Dataset,
    sample_dims: Sequence[str] = (SAMPLE_DIM_NAME,),
) -> Sequence[tf.Tensor]:
    """
    Apply standard descaling to a series of standard-scaled
    layers based on mean and standard deviation from a batch of data.

    Args:
        names: variable name in batch of each layer in layers
        layers: input tensors to be scaled by de-scaling layers
        batch: reference data for mean and standard deviation
    
    Returns:
        denormalized_layers: de-scaled tensors
    """
    return _fit_norm_layer(
        StandardDenormLayer,  # type: ignore
        names=names,
        layers=layers,
        batch=batch,
        sample_dims=sample_dims,
    )
