import dacite
import dataclasses
from typing import Any, List, Mapping, Sequence, Union, Type
import tensorflow as tf

from .layers import (
    LinearBlock,
    ResidualOutput,
    FieldOutput,
    FieldInput,
    RNNBlock,
    MLPBlock,
)


def get_architecture_cls(key: str):

    if key == "rnn":
        return RNNBlock
    elif key == "dense":
        return MLPBlock
    elif key == "linear":
        return LinearBlock
    else:
        raise KeyError(f"Unrecognized architecture provided: {key}")


@dataclasses.dataclass
class ArchitectureParams:
    """
        name: Underlying model architecture to use for the emulator.
            See `get_architecture_cls` for a list of supported layers.
        kwargs: keyword arguments to pass to the initialization
            of the architecture layer
    """

    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def instance(self):
        cls = get_architecture_cls(self.name)
        return cls(**self.kwargs)


@dataclasses.dataclass
class Config:

    """
    Microphysics emulator model builder

    Args:
        input_variables: names of all inputs to the model
        direct_out_variables: names of primary outputs of the model.
        residual_out_variables: names of outputs using a residual output,
            analogous to learning model tendencies instead of direct
            field-to-field prediction. Residual output variable maps to
            an associated input name to increment.
        architecture: parameters to underlying model prediction architecture
            to use for emulation.
        normalize_key: Normalization style to use for inputs/outputs.  Pass None
            to disable normalization
        selection_map: Subselection mapping for feature dimension of input/output
            variables to slices along the feature dimension
        tendency_outputs: Additional outputs (tendencies) to get from the
            residual outputs.  The mapping key should match a variable in
            residual_outputs and the value will be the new output variable
            name
        timestep_increment_sec: Time increment multiplier for the state-tendency
            update
    """

    input_variables: List[str] = dataclasses.field(default_factory=list)
    direct_out_variables: List[str] = dataclasses.field(default_factory=list)
    residual_out_variables: Mapping[str, str] = dataclasses.field(default_factory=dict)
    architecture: ArchitectureParams = dataclasses.field(
        default_factory=lambda: ArchitectureParams(name="linear")
    )
    normalize_key: str = "mean_std"
    selection_map: Mapping[str, slice] = dataclasses.field(default_factory=dict)
    tendency_outputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    timestep_increment_sec: int = 900

    @classmethod
    def from_dict(cls, dict_) -> "Config":
        return dacite.from_dict(cls, dict_, dacite.Config(strict=True))

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

    def _get_direct_outputs(self, sample_out):

        outputs = []

        for name, sample in zip(self.direct_out_variables, sample_out):
            out_ = FieldOutput(
                sample,
                denormalize=self.normalize_key,
                name=name,
                enforce_positive=True,
            )
            outputs.append(out_)

        return outputs

    def _get_residual_outputs(self, sample_out, net_output, residual_to_input_map):

        outputs = []
        tendencies = []

        for (name, in_state), sample in zip(residual_to_input_map.items(), sample_out):
            # incremented state field output
            res_out = ResidualOutput(
                sample,
                self.timestep_increment_sec,
                denormalize=self.normalize_key,
                name=name,
                enforce_positive=True,
            )

            out_ = res_out([in_state, net_output])
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
            sample_out: Sample out tensors for determining layer shapes and
                fitting denormalization layers if specifies
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
        arch_layer = self.architecture.instance(processed)
        outputs = self._get_direct_outputs(sample_direct_out)
        outputs += self._get_residual_outputs(
            sample_residual_out, arch_layer, residual_map
        )

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model
