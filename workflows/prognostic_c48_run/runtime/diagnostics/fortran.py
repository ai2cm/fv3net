from typing import Dict, Mapping, Sequence
import dataclasses
import os

import fv3config

from .time import FortranTimeConfig

FORTRAN_PHYSICS_MODULES = ["gfs_phys", "gfs_sfc"]


@dataclasses.dataclass
class FortranVariableNameSpec:
    """Names required to specify Fortran diagnostic variable.

    Attributes:
        module_name: Fortran module of diagnostic
        field_name: name of variable in Fortran code
        output_name: name to use for variable in output diagnostic file
    """

    module_name: str
    field_name: str
    output_name: str

    def in_physics_module(self) -> bool:
        return self.module_name in FORTRAN_PHYSICS_MODULES


@dataclasses.dataclass
class FortranFileConfig:
    """Configurations for Fortran diagnostics defined in diag_table to be converted to zarr

    Attributes:
        name: filename of the diagnostic. Must include .zarr suffix.
        chunks: mapping of dimension names to chunk sizes
        variables: sequence of FortranVariableNameSpecs
        times: time configuration. Only kinds 'interval', 'interval-average' or 'every'
            are allowed.
    """

    name: str
    chunks: Mapping[str, int]
    variables: Sequence[FortranVariableNameSpec] = ()
    times: FortranTimeConfig = dataclasses.field(
        default_factory=lambda: FortranTimeConfig()
    )

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def to_fv3config_diag_file_config(self) -> fv3config.DiagFileConfig:
        frequency_units = "seconds"
        frequency = self.times.to_frequency(units=frequency_units)
        field_configs = [
            fv3config.DiagFieldConfig(
                variable.module_name,
                variable.field_name,
                variable.output_name,
                reduction_method=self.times.reduction_method(),
            )
            for variable in self.variables
        ]
        name_without_ext = os.path.splitext(self.name)[0]
        return fv3config.DiagFileConfig(
            name_without_ext, frequency, frequency_units, field_configs
        )
