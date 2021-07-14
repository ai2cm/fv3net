import abc
import dataclasses
import dacite
import tensorflow as tf
from toolz.dicttoolz import keyfilter
from typing import Any, Dict, List, Sequence, Union

import yaml

from fv3fit.emulation.data import TransformConfig, nc_dir_to_tf_dataset
from fv3fit.emulation.layers.norm import MaxProfileStdNormLayer, MaxProfileStdDenormLayer


@dataclasses.dataclass
class ModelTypeConfig(abc.ABC):
    # todo: right now i only include names for layer identification
    input_variables: List[str]
    output_variables: List[str]
    normalize_inputs: bool = True

    @abc.abstractmethod
    def get_model(self, data_example, **model_kwargs) -> tf.keras.Model:
        pass


class _Register:

    def __init__(self) -> None:
        self._registered_objs = {}

    def __call__(self, cls):
        self._registered_objs[cls.__name__] = cls
        return cls

    def __getitem__(self, key):
        return self._registered_objs[key]


_model_register = _Register()


@_model_register
@dataclasses.dataclass
class MLPEmulator(ModelTypeConfig):
    depth: int = 2
    width: int = 250

    # Normalize layer should probably be its own config, check if fit
    # then give a layer.  Layer can just do data inputs -> model 
    # inputs outputs -> model outputs

    def get_model(self, data_example, **model_kwargs) -> tf.keras.Model:

        inputs_, outputs_ = data_example

        input_layers = [
            tf.keras.layers.Input(X.shape[-1], name=name)
            for name, X in zip(self.input_variables, inputs_)
        ]

        hidden_input = tf.concat(input_layers, axis=-1)
        for i in range(self.depth - 1):
            hidden_layer = tf.keras.layers.Dense(
                self.width, activation=tf.keras.activations.tanh
            )(hidden_input)
            hidden_input = hidden_layer
        outputs = [
            tf.keras.layers.Dense(y.shape[-1], name=name)(hidden_input)
            for name, y in zip(self.output_variables, outputs_)
        ]
        model = tf.keras.Model(inputs=input_layers, outputs=outputs)

        return model


def normalize_model_io(example_inputs, example_outputs, model_to_wrap: tf.keras.Model):

    inputs = model_to_wrap.inputs
    outputs = model_to_wrap.outputs

    new_inputs = []
    normalized_in = []
    for i, layer in enumerate(inputs):
        input_ = tf.keras.layers.Input(layer.shape[-1], name=layer.name)
        norm_ = MaxProfileStdNormLayer(name=f"normalized_{layer.name}")
        norm_.fit(example_inputs[i])
        norm_ = norm_(input_)

        new_inputs.append(input_)
        normalized_in.append(norm_)

    new_model = model_to_wrap(normalized_in)

    new_outputs = []
    for i, layer in enumerate(model_to_wrap.layers[-len(outputs):]):
        denorm_ = MaxProfileStdDenormLayer(name=f"denormalized_{layer.name}")
        denorm_.fit(example_outputs[i])
        if len(outputs) > 1:
            denorm_ = denorm_(new_model[i])
        else:
            denorm_ = denorm_(new_model)
        new_outputs.append(denorm_)

    return tf.keras.Model(inputs=new_inputs, outputs=new_outputs)


@dataclasses.dataclass
class EmulatorTrainConfig:
    model: "ModelTypeConfig"
    input_preprocessing: TransformConfig
    input_variables: Sequence[str]
    output_variables: Sequence[str]
    normalize_inputs: bool = True
    normalize_loss: bool = True
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-4

    @classmethod
    def from_dict(cls, kwargs):
        kwargs_copy: Dict[str, Union[Dict[str, Any], Any]] = dict(**kwargs)
        model_info = dict(**kwargs_copy)["model"]
        model_name = model_info["name"]
        model_kwargs = model_info["model_kwargs"]

        # Use dacite forward reference link to instantiate correct config
        forward_refs = {"ModelTypeConfig": _model_register[model_name]}
        model_kwargs_with_io = _maybe_update_io_vars(kwargs_copy, model_kwargs)
        kwargs_copy["model"] = model_kwargs_with_io

        input_config = kwargs_copy["input_preprocessing"]
        input_config_with_io = _maybe_update_io_vars(kwargs_copy, input_config)
        kwargs_copy["input_preprocessing"] = input_config_with_io
        
        return dacite.from_dict(
            data_class=cls,
            data=kwargs_copy,
            config=dacite.Config(forward_references=forward_refs)
        )
    

def _maybe_update_io_vars(parent: dict, child: dict):
    """Inject input/output variables into another dictionary if not present"""
    def is_io(key):
        return key in ["input_variables", "output_variables"]
    
    io_vars = keyfilter(is_io, parent)

    child = dict(**child)
    for k, v in io_vars.items():
        if k not in child:
            child[k] = v

    return child


if __name__ == "__main__":
    
    config_path = "/home/andrep/repos/fv3net/workflows/emulation/updated_train_config.yaml"
    with open(config_path, "r") as f:
        config_source = yaml.safe_load(f)
    
    config = EmulatorTrainConfig.from_dict(config_source)
    
    tf_ds = nc_dir_to_tf_dataset(
        "/mnt/disks/scratch/training-subsets/simple-phys-hybridedmf-10day",
        config.input_preprocessing
    )
    example_data = next(iter(tf_ds.batch(20_000)))
    X, y = example_data

    # begin of emulator operations
    model = config.model.get_model(example_data)
    normalized_model = normalize_model_io(X, y, model)
    normalized_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    normalized_model.fit(
        tf_ds.shuffle(10_000).batch(64),
        epochs=1
    )
    pass
