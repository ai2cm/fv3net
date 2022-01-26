import dataclasses
from typing import List, Union, Mapping
from fv3fit.emulation.layers import (
    FieldInput,
    FieldOutput,
    IncrementedFieldOutput,
    ArchitectureConfig,
)
from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.keras.adapters import ensure_dict_output
from fv3fit.emulation.transforms import TensorTransform
import tensorflow as tf

__all__ = ["TransformedModelConfig", "transform_model"]


@dataclasses.dataclass
class TransformedModelConfig:
    """Builds a model supporting transformed input/output variables

    The ML prediction has the following flow::

        (input data dict) ->
        transform ->
        scale ->
        embed ->
        architecture ->
        unscale ->
        untransform

        Attributes:
            fields: a list of input/output fields...inputs and outputs are
                inferred like this::

                    outputs = [field for field in fields if field.output_name]
                    inputs = [field for field in fields if field.input_name]

    """

    architecture: ArchitectureConfig
    fields: List[Field]
    timestep_increment_sec: int
    normalize_key: str = "mean_std"
    enforce_positive: bool = False

    @property
    def name(self) -> str:
        return "transformed-model"

    @property
    def input_variables(self) -> List[str]:
        """The input variables in transformed space"""
        return [field.input_name for field in self.fields if field.input_name]

    @property
    def output_variables(self) -> List[str]:
        """The output variables in transformed space"""
        return [field.output_name for field in self.fields if field.output_name]

    def build(self, data: Mapping[str, tf.Tensor],) -> tf.keras.Model:
        factory = FieldFactory(
            self.timestep_increment_sec, self.normalize_key, self.enforce_positive, data
        )
        model = InnerModel(self.fields, self.architecture, factory)
        return model


def build_field_output(
    field: Field,
    data: Mapping[str, tf.Tensor],
    dt_sec: int,
    normalize: str,
    enforce_positive: bool,
) -> Union[FieldInput, FieldOutput, IncrementedFieldOutput]:
    # need name to avoid clashing normalization data
    name = f"output_{field.output_name}"
    if field.residual:
        return IncrementedFieldOutput(
            dt_sec=dt_sec,
            sample_in=data[field.input_name],
            sample_out=data[field.output_name],
            denormalize=normalize,
            enforce_positive=enforce_positive,
            name=name,
        )
    else:
        return FieldOutput(
            sample_out=data[field.output_name],
            denormalize=normalize,
            enforce_positive=enforce_positive,
            name=name,
        )


def build_field_input(
    field: Field, data: Mapping[str, tf.Tensor], normalize: str,
) -> Union[FieldOutput, IncrementedFieldOutput]:
    # need name to avoid clashing normalization data
    name = f"input_{field.input_name}"
    return FieldInput(
        sample_in=data[field.input_name],
        normalize=normalize,
        selection=field.selection.slice,
        name=name,
    )


@dataclasses.dataclass
class FieldFactory:
    _dt_sec: int
    _normalize: str
    _enforce_positive: bool
    _data: Mapping[str, tf.Tensor]

    def build_input(self, field: Field) -> FieldInput:
        return build_field_input(field, self._data, self._normalize)

    def build_output(self, field: Field,) -> tf.keras.layers.Layer:
        return build_field_output(
            field, self._data, self._dt_sec, self._normalize, self._enforce_positive,
        )

    def build_architecture(
        self, config: ArchitectureConfig, output_variables: List[str],
    ) -> tf.keras.layers.Layer:
        output_features = {key: self._data[key].shape[-1] for key in output_variables}
        return config.build(output_features)


class InnerModel(tf.keras.layers.Layer):
    """An inner model containg ML-trainable weights"""

    def __init__(
        self,
        fields: List[Field],
        architecture_config: ArchitectureConfig,
        factory: FieldFactory,
    ):
        super().__init__()

        outputs = [field for field in fields if field.output_name]
        inputs = [field for field in fields if field.input_name]

        self.arch = factory.build_architecture(
            architecture_config,
            output_variables=[field.output_name for field in outputs],
        )

        self._inputs = {
            field.input_name: factory.build_input(field) for field in inputs
        }
        self.inputs = None

        self.outputs = {
            field.output_name: factory.build_output(field) for field in outputs
        }
        self.output_fields = {
            field.output_name: field for field in fields if field.output_name
        }

    def _process_outputs(
        self, data: Mapping[str, tf.Tensor], outputs: Mapping[str, tf.Tensor]
    ) -> Mapping[str, tf.Tensor]:
        processed_outputs = {}
        for key in self.outputs:
            field = self.output_fields[key]
            if field.residual:
                processed_outputs[key] = self.outputs[key](
                    data[field.input_name], outputs[field.output_name]
                )
            else:
                processed_outputs[key] = self.outputs[key](outputs[field.output_name])
        return processed_outputs

    @tf.function
    def call(self, data: Mapping[str, tf.Tensor]):
        processed_inputs = {key: self._inputs[key](data[key]) for key in self._inputs}
        outputs = self.arch(processed_inputs)
        processed_outputs = self._process_outputs(data, outputs)
        return processed_outputs


def transform_model(
    model: tf.keras.Model,
    transform: TensorTransform,
    inputs: Mapping[str, tf.keras.Input],
) -> tf.keras.Model:
    try:
        model = ensure_dict_output(model)
    except ValueError:
        pass
    # Wrap the custom model with a keras functional model for easier
    # serialization. Serialized models need to know their input/output
    # signatures. The keras "Functional" API makes this explicit, but custom
    # models subclasses "remember" their first inputs. Since ``data``
    # contains both inputs and outputs the serialized model will think its
    # outputs are also inputs and never be able to evaluate...even though
    # calling `model(data)` works just fine.
    outputs = model(transform.forward(inputs))
    outputs = transform.backward(outputs)
    functional_keras_model = tf.keras.Model(
        inputs=inputs, outputs=transform.backward(outputs)
    )
    return ensure_dict_output(functional_keras_model)
