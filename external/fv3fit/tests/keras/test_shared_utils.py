import numpy as np
import tensorflow as tf

from fv3fit.keras._models.shared.utils import standard_normalize, standard_denormalize


def get_names_layers_arrays():
    names = list("abc")
    sample = np.arange(20).reshape(10, 2).astype(np.float32)
    arrays = [sample] * 3
    layers = [tf.convert_to_tensor(sample)] * 3
    return names, layers, arrays


def test_standard_normalize(regtest):
    outputs = standard_normalize(*get_names_layers_arrays())
    print(outputs, file=regtest)


def test_standard_denormalize(regtest):
    outputs = standard_denormalize(*get_names_layers_arrays())
    print(outputs, file=regtest)
