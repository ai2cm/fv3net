import dataclasses
from typing import Any, List, Mapping
import tensorflow as tf

from .layers import ResidualOutput, FieldOutput, FieldInput, RNNBlock, MLPBlock


def get_architecture_cls(key):

    if key == "rnn":
        return RNNBlock
    elif key == "dense":
        return MLPBlock
    else:
        raise KeyError(f"Unrecognized architecture provided: {key}")


@dataclasses.dataclass
class MicrophysicsModel:

    input_variables: List[str]
    output_variables: List[str]
    architecture: str = "rnn"
    arch_params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    normalize_key: str = "mean_std"
    selection_map: Mapping[str, slice] = dataclasses.field(default_factory=dict)
    residual_to_state: Mapping[str, str] = dataclasses.field(default_factory=dict)
    timestep_increment_sec: int = 900

    def _get_processed_inputs(self, sample_in, inputs):

        inputs = [
            FieldInput(
                sample_in=sample,
                normalize=self.normalize_key,
                selection=self.selection_map.get(name, None),
                name=name,
            )(tensor)
            for name, sample, tensor in zip(self.input_variables, sample_in, inputs)
        ]

        return inputs

    def _get_outputs(self, sample_out, net_output, state_map_for_residuals):

        outputs = []
        for name, sample in zip(self.output_variables, sample_out):
            if name in state_map_for_residuals:
                out_ = ResidualOutput(sample, self.timestep_increment_sec, name=name)
                in_state = state_map_for_residuals[name]
                out_ = out_([in_state, net_output])
            else:
                out_ = FieldOutput(sample, name=name)
                out_ = out_(net_output)

            outputs.append(out_)

        return outputs

    def build(self, sample_in, sample_out):

        inputs = [
            tf.keras.layers.Input(sample.shape[-1], name=name)
            for name, sample in zip(self.input_variables, sample_in)
        ]
        processed = self._get_processed_inputs(sample_in, inputs)
        arch_layer = get_architecture_cls(self.architecture)(**self.arch_params)(
            processed
        )
        outputs = self._get_outputs(sample_out, arch_layer)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model
