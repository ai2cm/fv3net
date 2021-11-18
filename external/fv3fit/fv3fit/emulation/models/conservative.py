import dataclasses
from typing import List, Mapping, Optional, TypeVar
import tensorflow as tf

from fv3fit.emulation.thermo import mass_integrate

T = TypeVar("T")


# TODO move up a level...could be used by data too
@dataclasses.dataclass
class Names:
    cloud_water: str = "cloud_water_mixing_ratio_input"
    specific_humidity: str = "specific_humidity_input"
    pressure_thickness: str = "pressure_thickness"
    surface_precipitation: str = "surface_precipitation"

    def total_water(self, d: Mapping[str, T]) -> T:
        return d[self.cloud_water] + d[self.specific_humidity]


class ConservativeWaterModel(tf.keras.Model):
    def __init__(
        self,
        model: tf.keras.Model,
        input_names: Optional[Names],
        output_names: Optional[Names],
        vertical_axis: int = -1,
    ):
        super().__init__()
        self.model = model
        self._input_names = input_names or Names()
        self._output_names = output_names or Names(
            "cloud_water_mixing_ratio_output", "specific_humidity_output", "pressure"
        )
        self.vertical_axis = vertical_axis

    def call(self, in_: Mapping[str, tf.Tensor]) -> List[tf.Tensor]:
        in_dict = dict(zip(self.model.input_names, in_))
        out = self.model(in_)
        out_dict = dict(zip(self.model.output_names, out))

        water_before = self._input_names.total_water(in_dict)
        water_after = self._output_names.total_water(out_dict)
        pressure_thickness = out_dict[self.input_names.pressure_thickness]

        column_water_before = mass_integrate(
            water_before, pressure_thickness, axis=self.vertical_axis
        )
        column_water_after = mass_integrate(
            water_after, pressure_thickness, axis=self.vertical_axis
        )

        out_dict[self._output_names.surface_precipitation] = (
            column_water_before - column_water_after
        )
        return [out_dict[k] for k in sorted(out_dict)]
