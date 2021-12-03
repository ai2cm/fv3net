import dataclasses
from typing import List, Mapping

import dacite
import tensorflow as tf
from fv3fit._shared import SliceConfig

from ..layers import FieldInput, FieldOutput, IncrementedFieldOutput
from ._core import ArchitectureConfig, get_combine_from_arch_key

from fv3fit.emulation import thermo


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
        enforce_positive: Enforce model outputs are zero or positive

    """

    input_variables: List[str] = dataclasses.field(default_factory=list)
    direct_out_variables: List[str] = dataclasses.field(default_factory=list)
    residual_out_variables: Mapping[str, str] = dataclasses.field(default_factory=dict)
    architecture: ArchitectureConfig = dataclasses.field(
        default_factory=lambda: ArchitectureConfig(name="linear")
    )
    normalize_key: str = "mean_std"
    selection_map: Mapping[str, SliceConfig] = dataclasses.field(default_factory=dict)
    tendency_outputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    timestep_increment_sec: int = 900
    enforce_positive: bool = True

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
        return self.direct_out_variables + list(self.residual_out_variables.keys())

    def _get_processed_inputs(self, sample_in, inputs):
        return {
            name: FieldInput(
                sample_in=sample_in[name],
                normalize=self.normalize_key,
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
                sample.shape[-1],
                sample_out=sample,
                denormalize=self.normalize_key,
                name=name,
                enforce_positive=self.enforce_positive,
            )(net_output)
            outputs[name] = out_
        return outputs

    def _get_residual_outputs(self, inputs, data, net_output):

        outputs = {}
        for name in self.residual_out_variables:
            # incremented state field output
            in_state = inputs[self.residual_out_variables[name]]
            sample = data[name]
            res_out = IncrementedFieldOutput(
                sample.shape[-1],
                self.timestep_increment_sec,
                sample_out=sample,
                denormalize=self.normalize_key,
                name=name,
                enforce_positive=self.enforce_positive,
            )
            out_ = res_out(in_state, net_output)
            outputs[name] = out_

            if name in self.tendency_outputs:
                tend_name = self.tendency_outputs[name]
                tendency = res_out.get_tendency_output(net_output)
                outputs[tend_name] = tendency

        return outputs

    def _compute_hidden(self, inputs, data):
        processed = self._get_processed_inputs(data, inputs)
        combine_layer = get_combine_from_arch_key(self.architecture.name)
        combined = combine_layer(processed)
        arch_layer = self.architecture.build()
        return arch_layer(combined)

    def _get_inputs(self, data):
        return {
            name: tf.keras.layers.Input(data[name].shape[-1], name=name)
            for name in self.input_variables
        }

    def build(self, data: Mapping[str, tf.Tensor],) -> tf.keras.Model:
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
                **self._get_residual_outputs(inputs, data, hidden),
            },
        )


@dataclasses.dataclass(frozen=True)
class Field:
    """Configuration describing a prognostic field

    A field with no ``output_name`` can be interpreted as an ML input only.
    A field with no ``input_name`` is a diagnostic.

    Attributes:
        output_name: the name of the variable representing a field after being
            modified by the an emulator
        input_name: the name of the output variable
        tendency_name: the name to call the output-input difference by
        selection: how to subset the feature-space of the field
    
    Example:

        >>> air_temperature = Field(
        ...     output_name="air_temperature_after_step",
        ...     input_name="air_temperature_before_step",
        ...     tendency_name="tendency_of_air_temperature_due_to_step",
        ...)

    """

    output_name: str = ""
    input_name: str = ""
    residual: bool = True
    # only used if residual is True
    tendency_name: str = ""
    selection: SliceConfig = dataclasses.field(default_factory=SliceConfig)


@dataclasses.dataclass(frozen=True)
class ZhaoCarrFields:
    """Relationship between names and the physical input/outputs of the
    Zhao-carr microphysics
    """

    cloud_water: Field = Field(
        "cloud_water_mixing_ratio_after_precpd",
        "cloud_water_mixing_ratio_input",
        residual=False,
    )

    specific_humidity: Field = Field(
        "specific_humidity_after_precpd", "specific_humidity_input", residual=True,
    )

    air_temperature: Field = Field(
        "air_temperature_after_precpd", "air_temperature_input", residual=True,
    )
    surface_precipitation: Field = Field(output_name="total_precipitation")
    pressure_thickness = Field(input_name="pressure_thickness_of_atmospheric_layer")


@dataclasses.dataclass
class ConservativeWaterConfig:
    """Builds a model that diagnoses surface precipitation based on the total
    water sink predicted by another ML model


    The model returned by .build will include this surface precipitation
    under the output name ``fields.surface_precipitation``.

    Attributes:
        extra_input_variables: extra inputs beyond cloud water, specific
            humidity, and air temperature to pass to the ML architecture.
    """

    fields: ZhaoCarrFields = ZhaoCarrFields()
    architecture: ArchitectureConfig = dataclasses.field(
        default_factory=lambda: ArchitectureConfig(name="linear")
    )
    extra_input_variables: List[Field] = dataclasses.field(default_factory=list)
    normalize_key: str = "mean_std"
    timestep_increment_sec: int = 900
    enforce_positive: bool = True

    @property
    def _prognostic_fields(self) -> List[Field]:
        return [
            self.fields.cloud_water,
            self.fields.air_temperature,
            self.fields.specific_humidity,
        ]

    def _build_base_model(self, data) -> tf.keras.Model:

        prognostic_variables = self._prognostic_fields

        return MicrophysicsConfig(
            input_variables=[
                v.input_name for v in prognostic_variables + self.extra_input_variables
            ],
            direct_out_variables=[
                var.output_name for var in prognostic_variables if not var.residual
            ],
            residual_out_variables={
                var.output_name: var.input_name
                for var in prognostic_variables
                if var.residual
            },
            tendency_outputs={
                var.output_name: var.tendency_name
                for var in prognostic_variables
                if var.residual and var.tendency_name
            },
            architecture=self.architecture,
            normalize_key=self.normalize_key,
            timestep_increment_sec=self.timestep_increment_sec,
            enforce_positive=self.enforce_positive,
            selection_map={v.input_name: v.selection for v in self._input_variables},
        ).build(data)

    @property
    def _input_variables(self) -> List[Field]:
        return (
            [v for v in self._prognostic_fields]
            + [self.fields.pressure_thickness]
            + self.extra_input_variables
        )

    @property
    def input_variables(self) -> List[str]:
        return [v.input_name for v in self._input_variables]

    @property
    def output_variables(self) -> List[str]:
        return [v.output_name for v in self._prognostic_fields] + [
            self.fields.surface_precipitation.output_name
        ]

    @property
    def name(self):
        return f"conservative-microphysics-emulator-{self.architecture.name}"

    def build(self, data: Mapping[str, tf.Tensor]) -> tf.keras.Model:
        model = self._build_base_model(data)
        return _assoc_conservative_precipitation(model, self.fields)


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

    inputs = dict(zip(model.input_names, model.inputs))
    nz = inputs[fields.cloud_water.input_name].shape[-1]
    inputs[fields.pressure_thickness.input_name] = tf.keras.Input(
        shape=[nz], name=fields.pressure_thickness.input_name
    )

    out = model(inputs)
    out[
        fields.surface_precipitation.output_name
    ] = thermo.conservative_precipitation_zhao_carr(
        specific_humidity_before=inputs[fields.specific_humidity.input_name],
        specific_humidity_after=out[fields.specific_humidity.output_name],
        cloud_before=inputs[fields.cloud_water.input_name],
        cloud_after=out[fields.cloud_water.output_name],
        mass=thermo.layer_mass(inputs[fields.pressure_thickness.input_name]),
    )
    new_model = tf.keras.Model(inputs=inputs, outputs=out)
    new_model(inputs)
    return new_model
