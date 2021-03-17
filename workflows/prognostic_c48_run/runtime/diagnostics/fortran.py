from typing import Dict, Mapping, Optional, Sequence
import dataclasses
import datetime
import os

import cftime
import fv3config

from .time import All, IntervalTimes, TimeContainer

# the Fortran model handles diagnostics from these modules in a special way
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
class FortranTimeConfig:
    """Configuration for output times from Fortran diagnostics.

    Attributes:
        kind: one of "interval", "interval-average" or "every"
        frequency: frequency in seconds, used for "interval" or "interval-average"

    Note:
        Fortran diagnostics from the "gfs_phys" or "gfs_sfc" modules must all use the
        same output interval frequency. Furthermore, these outputs will be the same
        whether the time kind is ``interval`` or ``interval-average`` since their
        time-averaging is handled by the GFS physics code instead of the FMS
        diagnostics manager.
    """

    kind: str = "every"
    frequency: Optional[float] = None

    def time_container(self, initial_time: cftime.DatetimeJulian) -> TimeContainer:
        if self.kind == "interval" and self.frequency:
            return TimeContainer(IntervalTimes(self.frequency, initial_time))
        elif self.kind == "every":
            return TimeContainer(All())
        elif self.kind == "interval-average" and self.frequency:
            return IntervalAveragedTimes(
                datetime.timedelta(seconds=self.frequency), initial_time, False,
            )
        else:
            raise NotImplementedError(f"Time {self.kind} not implemented.")

    def to_frequency(self, units="seconds") -> datetime.timedelta:
        if self.kind == "every":
            return datetime.timedelta(seconds=0)
        elif self.kind.startswith("interval") and self.frequency:
            return datetime.timedelta(seconds=self.frequency)
        else:
            raise NotImplementedError(f"Time {self.kind} not implemented.")

    def reduction_method(self) -> str:
        if self.kind == "interval-average":
            return "average"
        else:
            return "none"


@dataclasses.dataclass
class FortranFileConfig:
    """Configurations for Fortran diagnostics defined in diag_table to be converted to zarr

    Attributes:
        name: filename of the diagnostic. Must include .zarr suffix.
        chunks: mapping of dimension names to chunk sizes
        variables: sequence of FortranVariableNameSpecs
        times: time configuration. Only kinds 'interval', 'interval-average' or 'every'
            are allowed.

    Example:
        name: dycore_diags.zarr
        chunks:
            time: 96
        times:
            kind: interval
            frequency: 900
        variables:
            - {module_name: "dynamics", field_name: "tq", output_name: "PWAT"}
            - {module_name: "dynamics", field_name: "z500", output_name: "h500"}
            - {module_name: "dynamics", field_name: "tb", output_name: "TMPlowest"}
            - {module_name: "dynamics", field_name: "t850", output_name: "TMP850"}
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
        frequency = self.times.to_frequency().total_seconds()
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


def file_configs_to_namelist_settings(
    diagnostics: Sequence[FortranFileConfig], physics_timestep: datetime.timedelta
) -> Mapping[str, Mapping]:
    """Return overlay for physics output frequency configuration if any physics
    diagnostics are specified in given sequence of FortranFileConfig's."""
    physics_frequencies = set()
    for diagnostic_config in diagnostics:
        for variable in diagnostic_config.variables:
            if variable.in_physics_module():
                variable_frequency = diagnostic_config.times.to_frequency()
                physics_frequencies.add(variable_frequency)

    if len(physics_frequencies) == 0:
        return {}
    elif len(physics_frequencies) == 1:
        physics_frequency = list(physics_frequencies)[0]
        if physics_frequency == datetime.timedelta(seconds=0):
            # handle case of outputting diagnostics on every physics timestep
            physics_frequency = physics_timestep
        one_hour_duration = datetime.timedelta(hours=1)
        return {
            "namelist": {
                "atmos_model_nml": {"fhout": physics_frequency / one_hour_duration},
                "gfs_physics_nml": {"fhzero": physics_frequency / one_hour_duration},
            }
        }
    else:
        raise NotImplementedError(
            "Cannot output physics diagnostics at multiple frequencies."
        )
