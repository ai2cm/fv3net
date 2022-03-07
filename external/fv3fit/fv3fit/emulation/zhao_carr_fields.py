from fv3fit.emulation.data.config import SliceConfig
import dataclasses


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
