import dataclasses
from typing import List, Union, Mapping
from fv3fit.emulation.layers import (
    FieldInput,
    FieldOutput,
    IncrementedFieldOutput,
    ArchitectureConfig,
)
from fv3fit.emulation.zhao_carr_fields import Field
from fv3fit.emulation import transforms
import tensorflow as tf

__all__ = ["TransformedModelConfig"]


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

    architecture_config: ArchitectureConfig
    fields: List[Field]
    dt_sec: int
    normalize: str = "mean_std"
    enforce_positive: bool = False

    @property
    def name(self) -> str:
        return "transformed-model"

    @property
    def input_variables(self) -> List[str]:
        return [field.input_name for field in self.fields if field.input_name]

    @property
    def output_variables(self) -> List[str]:
        return [field.output_name for field in self.fields if field.output_name]

    def build(
        self,
        data: Mapping[str, tf.Tensor],
        transform: transforms.TensorTransform = transforms.Identity,
    ) -> "TransformedModel":
        factory = FieldFactory(self.dt_sec, self.normalize, self.enforce_positive, data)
        return TransformedModel(
            self.fields, self.architecture_config, factory, transform=transform
        )


def build_field_output(
    field: Field,
    data: Mapping[str, tf.Tensor],
    dt_sec: int,
    normalize: str,
    enforce_positive: bool,
) -> Union[FieldInput, FieldOutput, IncrementedFieldOutput]:
    if field.residual:
        return IncrementedFieldOutput(
            dt_sec=dt_sec,
            sample_in=data[field.input_name],
            sample_out=data[field.output_name],
            denormalize=normalize,
            enforce_positive=enforce_positive,
        )
    else:
        return FieldOutput(
            sample_out=data[field.output_name],
            denormalize=normalize,
            enforce_positive=enforce_positive,
        )


def build_field_input(
    field: Field, data: Mapping[str, tf.Tensor], normalize: str,
) -> Union[FieldOutput, IncrementedFieldOutput]:
    return FieldInput(
        sample_in=data[field.input_name],
        normalize=normalize,
        selection=field.selection.slice,
    )


@dataclasses.dataclass
class FieldFactory:
    _dt_sec: int
    _normalize: str
    _enforce_positive: bool
    _data: Mapping[str, tf.Tensor]

    def build_input(
        self, field: Field, transform: transforms.TensorTransform
    ) -> FieldInput:
        return build_field_input(field, transform.forward(self._data), self._normalize)

    def build_output(
        self, field: Field, transform: transforms.TensorTransform
    ) -> tf.keras.layers.Layer:
        return build_field_output(
            field,
            transform.forward(self._data),
            self._dt_sec,
            self._normalize,
            self._enforce_positive,
        )

    def build_architecture(
        self,
        config: ArchitectureConfig,
        output_variables: List[str],
        transform: transforms.TensorTransform,
    ) -> tf.keras.layers.Layer:
        data = transform.forward(self._data)
        output_features = {key: data[key].shape[-1] for key in output_variables}
        return config.build(output_features)


class TransformedModel(tf.keras.Model):
    def __init__(
        self,
        fields: List[Field],
        architecture_config: ArchitectureConfig,
        factory: FieldFactory,
        transform: transforms.TensorTransform = transforms.Identity,
    ):
        super().__init__()
        self.transform = transform

        outputs = [field for field in fields if field.output_name]
        inputs = [field for field in fields if field.input_name]

        self.arch = factory.build_architecture(
            architecture_config,
            output_variables=[field.output_name for field in outputs],
            transform=transform,
        )

        self.inputs = {
            field.input_name: factory.build_input(field, transform) for field in inputs
        }

        self.outputs = {
            field.output_name: factory.build_output(field, transform)
            for field in outputs
        }

    @tf.function
    def call(self, data: Mapping[str, tf.Tensor]):
        data = self.transform.forward(data)
        processed_inputs = {key: self.inputs[key](data[key]) for key in self.inputs}
        outputs = self.arch(processed_inputs)
        processed_outputs = {key: outputs[key] for key in self.outputs}
        return self.transform.backward(processed_outputs)
