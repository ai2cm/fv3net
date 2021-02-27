from typing import Dict, Mapping
import dataclasses

import fv3config

from .time import TimeConfig

# keys are names used for diagnostic in output netCDFs files for typical diag_tables
# and values are a tuple of:
# MODULE_NAME (Fortran module that diagnostic is in)
# FIELD_NAME (name of diagnostic in Fortran code)
MODULE_FIELD_NAME_TABLE = {
    "h500": ("dynamics", "z500"),
    "PRATEsfc": ("gfsphys", "totprcpb_ave"),
}


@dataclasses.dataclass
class FortranFileConfig:
    """Configurations for Fortran diagnostics defined in diag_table to be converted to zarr

    Attributes:
        name: filename of the diagnostic. Must include .zarr suffix. For example, if
            atmos_8xdaily is defined in diag_table, use atmos_8xdaily.zarr here.
        chunks: mapping of dimension names to chunk sizes
    """

    name: str
    variables: Sequence[str] = []
    times: TimeConfig = dataclasses.field(default_factory=lambda: TimeConfig())
    chunks: Mapping[str, int]

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def to_fv3config_diag_file_config(self) -> fv3config.DiagFileConfig:
        if self.times.kind in ["interval", "interval-average"]:
            frequency = self.times.frequency
            frequency_units = "seconds"
        elif self.times.kind == "every":
            frequency = 0
            frequency_units = "seconds"
        else:
            raise NotImplementedError(
                "Fortran diagnostics can only use a times 'kind' that is one of "
                "'interval', 'interval-average' or 'every'."
            )

        if self.times.kind == "interval-average":
            reduction_method = "average"
        else:
            reduction_method = "none"

        field_configs = [
            _field_config_from_variable(variable, reduction_method)
            for variable in self.variables
        ]

        return fv3config.DiagFileConfig(
            self.name, frequency, frequency_units, field_configs
        )

    @staticmethod
    def _field_config_from_variable(
        output_name: str, reduction_method: str
    ) -> fv3config.DiagFieldConfig:
        module_name, field_name = MODULE_FIELD_NAME_TABLE[output_name]
        return fv3config.DiagFieldConfig(
            module_name, field_name, output_name, reduction_method=reduction_method
        )

