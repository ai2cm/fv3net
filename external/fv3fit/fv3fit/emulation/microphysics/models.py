import dacite
import dataclasses
from typing import Any, List, Mapping, Sequence
import tensorflow as tf

from .layers import (
    CombineInputs,
    IncrementedFieldOutput,
    FieldOutput,
    FieldInput,
    RNNBlock,
    MLPBlock,
)


def get_architecture_cls(key: str, kwargs: Mapping[str, Any]):

    if key == "rnn":
        return RNNBlock(**kwargs)
    elif key == "dense":
        return MLPBlock(**kwargs)
    elif key == "linear":
        return MLPBlock(depth=0)
    else:
        raise KeyError(f"Unrecognized architecture provided: {key}")


def get_combine_from_arch_key(key: str):

    if key == "rnn":
        return CombineInputs(-1, expand_axis=-1)
    else:
        return CombineInputs(-1, expand_axis=None)


@dataclasses.dataclass
class ArchitectureConfig:
    """
        name: Name of underlying model architecture to use for the emulator.
            See `get_architecture_cls` for a list of supported layers.
        kwargs: keyword arguments to pass to the initialization
            of the architecture layer
    """

    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def build(self):
        return get_architecture_cls(self.name, kwargs=self.kwargs)


@dataclasses.dataclass
class Config:

    """
    Microphysics emulator model builder

    Args:
        input_variables: names of all inputs to the model
        direct_out_variables: names of direct field prediction outputs of the model.
        residual_out_variables: names of outputs using a residual-based output,
            analogous to learning model tendencies instead of direct
            field-to-field prediction. The mapping of residual output variable names
            to an associated model input name to increment. E.g.,
            {"air_temperature_output": "air_temperature_input"} produces the
            output air_temperature_output = air_temperature_input + tendency * timestep
        architecture: `ArchitectureConfig` object initialized with keyword
            arguments "name" (key for architecture layer) and "kwargs" (mapping
            of any keyword arguments to initialize the layer)
        normalize_key: Normalization style to use for inputs/outputs.  Pass None
            to disable normalization
        selection_map: Subselection mapping for feature dimension of input/output
            variables to slices along the feature dimension
        tendency_outputs: Additional output tendencies to get from the
            residual-based output layers.  The mapping key should match a variable in
            residual_outputs and the value will be the new output variable
            name.
        timestep_increment_sec: Time increment multiplier for the state-tendency
            update
    """

    input_variables: List[str] = dataclasses.field(default_factory=list)
    direct_out_variables: List[str] = dataclasses.field(default_factory=list)
    residual_out_variables: Mapping[str, str] = dataclasses.field(default_factory=dict)
    architecture: ArchitectureConfig = dataclasses.field(
        default_factory=lambda: ArchitectureConfig(name="linear")
    )
    normalize_key: str = "mean_std"
    selection_map: Mapping[str, slice] = dataclasses.field(default_factory=dict)
    tendency_outputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    timestep_increment_sec: int = 900

    @classmethod
    def from_dict(cls, dict_) -> "Config":
        return dacite.from_dict(cls, dict_, dacite.Config(strict=True))

    @property
    def output_variables(self):
        return (
            self.direct_out_variables
            + list(self.residual_out_variables.keys())
            + list(self.tendency_outputs.values())
        )

    def _get_processed_inputs(self, sample_in, inputs):

        inputs = [
            FieldInput(
                sample_in=sample,
                normalize=self.normalize_key,
                selection=self.selection_map.get(name, None),
                name=f"processed_{name}",
            )(tensor)
            for name, sample, tensor in zip(self.input_variables, sample_in, inputs)
        ]

        return inputs

    def _get_direct_outputs(self, sample_out, net_output):

        outputs = []

        for name, sample in zip(self.direct_out_variables, sample_out):
            out_ = FieldOutput(
                sample,
                denormalize=self.normalize_key,
                name=name,
                enforce_positive=True,
            )(net_output)
            outputs.append(out_)

        return outputs

    def _get_residual_outputs(self, sample_out, net_output, residual_to_input_map):

        outputs = []
        tendencies = []

        for (name, in_state), sample in zip(residual_to_input_map.items(), sample_out):
            # incremented state field output
            res_out = IncrementedFieldOutput(
                sample,
                self.timestep_increment_sec,
                denormalize=self.normalize_key,
                name=name,
                enforce_positive=True,
            )

            out_ = res_out(in_state, net_output)
            outputs.append(out_)

            if name in self.tendency_outputs:
                tend_name = self.tendency_outputs[name]
                tendency = res_out.get_tendency_output(net_output)
                renamed = tf.keras.layers.Lambda(lambda x: x, name=tend_name)(tendency)
                tendencies.append(renamed)

        return outputs + tendencies

    def build(
        self,
        sample_in: Sequence[tf.Tensor],
        sample_direct_out: Sequence[tf.Tensor] = None,
        sample_residual_out: Sequence[tf.Tensor] = None,
    ):
        """
        Build model described by the configuration

        Args:
            sample_in: Sample input tensors for determining layer shapes and
                fitting normalization layers if specifies
            sample_direct_out: Sample of direct output field tensors used to
                fit denormalization layers (if specified) and shape
            sample_residual_out: Sample of residual output field tensors used to
                fit denormalization layers (if specified) and shape.  Note: the
                samples should be in tendency form to produce proper denormalization
        """

        if sample_direct_out is None:
            sample_direct_out = []

        if sample_residual_out is None:
            sample_residual_out = []

        inputs = [
            tf.keras.layers.Input(sample.shape[-1], name=name)
            for name, sample in zip(self.input_variables, sample_in)
        ]
        residual_map = {
            resid_name: inputs[self.input_variables.index(input_name)]
            for resid_name, input_name in self.residual_out_variables.items()
        }
        processed = self._get_processed_inputs(sample_in, inputs)
        combine_layer = get_combine_from_arch_key(self.architecture.name)
        combined = combine_layer(processed)
        arch_layer = self.architecture.build
        arch_out = arch_layer(combined)
        outputs = self._get_direct_outputs(sample_direct_out, arch_out)
        outputs += self._get_residual_outputs(
            sample_residual_out, arch_out, residual_map
        )

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model
