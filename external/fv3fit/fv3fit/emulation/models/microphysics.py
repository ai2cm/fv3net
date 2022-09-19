import dataclasses
import dacite
import tensorflow as tf
from typing import List, Mapping, Optional
import vcm

from fv3fit._shared import SliceConfig
from fv3fit.emulation import thermo
from fv3fit.emulation.layers.normalization2 import (
    NormFactory,
    StdDevMethod,
    MeanMethod,
)
from fv3fit.keras.adapters import ensure_dict_output
from fv3fit.emulation.zhao_carr_fields import ZhaoCarrFields
from fv3fit.emulation.models.base import Model
from fv3fit.emulation.layers import (
    FieldInput,
    FieldOutput,
    ArchitectureConfig,
)


@dataclasses.dataclass
class MicrophysicsConfig:

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
        unscaled_outputs: outputs that won't be rescaled.
            ``direct_out_variables`` are rescaled before being returned by the
            constructed model.
        architecture: `ArchitectureConfig` object initialized with keyword
            arguments "name" (key for architecture layer) and "kwargs" (mapping
            of any keyword arguments to initialize the layer)
        normalize_default: default normalization to use for inputs/outputs
        normalize_map: overrides the default normalization per variable
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
    normalize_default: Optional[NormFactory] = NormFactory(
        scale=StdDevMethod.all, center=MeanMethod.per_feature
    )
    selection_map: Mapping[str, SliceConfig] = dataclasses.field(default_factory=dict)
    normalize_map: Mapping[str, NormFactory] = dataclasses.field(default_factory=dict)
    tendency_outputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    timestep_increment_sec: int = 900
    unscaled_outputs: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, d) -> "MicrophysicsConfig":
        return dacite.from_dict(cls, d, dacite.Config(strict=True))

    def asdict(self):
        return dataclasses.asdict(self)

    def __post_init__(self):
        self.selection_map_slices = {k: v.slice for k, v in self.selection_map.items()}

    @property
    def name(self) -> str:
        return f"microphysics-emulator-{self.architecture.name}"

    @property
    def output_variables(self) -> List[str]:
        return (
            self.direct_out_variables
            + list(self.residual_out_variables.keys())
            + list(self.tendency_outputs.values())
            + list(self.unscaled_outputs)
        )

    def _get_norm_factory(self, name: str) -> Optional[NormFactory]:
        return self.normalize_map.get(name, self.normalize_default)

    def _get_processed_inputs(self, sample_in, inputs):
        return {
            name: FieldInput(
                sample_in=sample_in[name],
                normalize=self._get_norm_factory(name),
                selection=self.selection_map_slices.get(name, None),
                name=f"processed_{name}",
            )(tensor)
            for name, tensor in inputs.items()
        }

    def _get_direct_outputs(self, data, net_output):

        outputs = {}

        for name in self.direct_out_variables:
            sample = data[name]
            out_ = FieldOutput(
                sample_out=sample, denormalize=self._get_norm_factory(name), name=name,
            )(net_output[name])
            outputs[name] = out_
        return outputs

    def _get_unscaled_outputs(self, net_output):
        return {
            name: tf.keras.layers.Lambda(lambda x: x, name=name)(net_output[name])
            for name in self.unscaled_outputs
        }

    def _compute_hidden(self, inputs, data):
        processed = self._get_processed_inputs(data, inputs)
        output_features = {key: data[key].shape[-1] for key in self.output_variables}
        arch_layer = self.architecture.build(output_features)
        return arch_layer(processed)

    def _get_inputs(self, data):
        return {
            name: tf.keras.layers.Input(data[name].shape[-1], name=name)
            for name in self.input_variables
        }

    def build(self, data: Mapping[str, tf.Tensor]) -> tf.keras.Model:
        """
        Build model described by the configuration

        Args:
            data: Sample input tensors for determining layer shapes and
                fitting normalization layers if specifies
        """
        inputs = self._get_inputs(data)
        hidden = self._compute_hidden(inputs, data)
        return tf.keras.models.Model(
            inputs=inputs,
            outputs={
                **self._get_direct_outputs(data, hidden),
                **self._get_unscaled_outputs(hidden),
            },
        )


def _check_types():
    # add some type assertions to enforce ensure that the model classes match
    # the protocol
    _: Model = MicrophysicsConfig()


def _assoc_conservative_precipitation(
    model: tf.keras.Model, fields: ZhaoCarrFields
) -> tf.keras.Model:
    """add conservative precipitation output to a model

    Args:
        model: a ML model
        fields: a description of how physics variables map onto the names of
            ``model`` and the data.

    Returns:
        a model with surface precipitation stored at
        ``fields.surface_precipitation.output_name``.

    """
    model = ensure_dict_output(model)
    inputs = dict(zip(model.input_names, model.inputs))
    nz = inputs[fields.cloud_water.input_name].shape[-1]
    inputs[fields.pressure_thickness.input_name] = tf.keras.Input(
        shape=[nz], name=fields.pressure_thickness.input_name
    )

    out = model(inputs)
    precip_scalar = thermo.conservative_precipitation_zhao_carr(
        specific_humidity_before=inputs[fields.specific_humidity.input_name],
        specific_humidity_after=out[fields.specific_humidity.output_name],
        cloud_before=inputs[fields.cloud_water.input_name],
        cloud_after=out[fields.cloud_water.output_name],
        mass=vcm.layer_mass(inputs[fields.pressure_thickness.input_name]),
    )
    out[fields.surface_precipitation.output_name] = precip_scalar[:, None]
    # convert_to_dict_output ensures that output names are consistent
    new_model = ensure_dict_output(tf.keras.Model(inputs=inputs, outputs=out))
    new_model(inputs)
    return new_model
