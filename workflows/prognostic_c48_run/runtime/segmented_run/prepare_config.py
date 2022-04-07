import dataclasses
import argparse
import emulation
import yaml
import logging
import sys
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence, Optional, TypeVar, Union

import dacite

import fv3config
import fv3kube
from fv3kube import RestartCategoriesConfig

import pandas as pd

from runtime.diagnostics.manager import FortranFileConfig
from runtime.diagnostics.fortran import file_configs_to_namelist_settings
from runtime.config import UserConfig


__all__ = ["to_fv3config", "InitialCondition", "FV3Config", "HighLevelConfig"]


logger = logging.getLogger(__name__)

PROGNOSTIC_DIAG_TABLE = "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"
SUPPRESS_RANGE_WARNINGS = {"namelist": {"fv_core_nml": {"range_warn": False}}}


def default_coupler_nml():
    return {"coupler_nml": {"dt_atmos": 900}}


T = TypeVar("T")


def instantiate_dataclass_from(cls: Callable[..., T], instance: Any) -> T:
    """Create an instance of ``cls`` with the same attributes as ``instance``

    This is useful for instantiating parent class from child classes. ``cls``
    must be a ``dataclass``.
    """
    fields = dataclasses.fields(cls)
    return cls(**{field.name: getattr(instance, field.name) for field in fields})


@dataclasses.dataclass
class FV3Config:
    """Dataclass representation of all fv3config fields **not** overriden by
    ``HighLevelConfig``
    """

    namelist: Any = dataclasses.field(default_factory=default_coupler_nml)
    experiment_name: Any = None
    data_table: Any = None
    field_table: Optional[str] = None
    forcing: Any = None
    orographic_forcing: Any = None
    gfs_analysis_data: Any = None

    def asdict(self):
        dict_ = dataclasses.asdict(self)
        return {k: v for k, v in dict_.items() if v is not None}


@dataclasses.dataclass
class InitialCondition:
    """An initial condition with the format ``{base_url}/{timestep}``

    Attributes:
        base_url: a location in GCS or local
        timestep: a YYYYMMDD.HHMMSS timestamp
        restart_categories: an optional mapping from the FV3GFS restart
            categories of 'core', 'surface', 'tracer' and 'surface_wind' to
            restart category names as stored on disk
    """

    base_url: str
    timestep: str
    restart_categories: RestartCategoriesConfig = dataclasses.field(
        default_factory=RestartCategoriesConfig
    )

    @property
    def overlay(self):
        return fv3kube.c48_initial_conditions_overlay(
            self.base_url, self.timestep, restart_categories=self.restart_categories
        )


@dataclasses.dataclass
class HighLevelConfig(UserConfig, FV3Config):
    """A high level configuration object for prognostic runs

    Combines fv3config and runtime configurations with conveniences for commonly
    used initial conditions and configurations

    Attributes:
        base_version: the default physics config
        initial_conditions: Specification for the initial conditions
        duration: the duration of each segment. If provided, overrides any
            settings in ``namelist.coupler_nml``. Must be a valid input to
            ``pandas.TimeDelta``.  Examples: ``3h``.

    See :py:class:`runtime.config.UserConfig` and :py:class:`FV3Config` for
    documentation on other allowed attributes

    """

    base_version: str = "v0.5"
    initial_conditions: Union[InitialCondition, Any] = ""
    duration: str = ""
    zhao_carr_emulation: emulation.EmulationConfig = emulation.EmulationConfig()

    @staticmethod
    def from_dict(dict_: dict) -> "HighLevelConfig":
        return dacite.from_dict(
            HighLevelConfig,
            dict_,
            dacite.Config(
                strict=True,
                type_hooks={
                    emulation.EmulationConfig: emulation.EmulationConfig.from_dict
                },
            ),
        )

    def _initial_condition_overlay(self):
        return (
            self.initial_conditions.overlay
            if isinstance(self.initial_conditions, InitialCondition)
            else {"initial_conditions": self.initial_conditions}
        )

    def _to_fv3config_specific(self) -> FV3Config:
        return FV3Config(
            namelist=self.namelist,
            experiment_name=self.experiment_name,
            data_table=self.data_table,
            field_table=self.field_table,
            forcing=self.forcing,
            orographic_forcing=self.orographic_forcing,
            gfs_analysis_data=self.gfs_analysis_data,
        )

    def to_runtime_config(self) -> UserConfig:
        """Extract just the python runtime configurations"""
        return instantiate_dataclass_from(UserConfig, self)

    def _physics_timestep(self) -> timedelta:
        dict_ = dataclasses.asdict(self._to_fv3config_specific())
        return timedelta(
            seconds=fv3kube.merge_fv3config_overlays(
                fv3kube.get_base_fv3config(self.base_version), dict_
            )["namelist"]["coupler_nml"]["dt_atmos"]
        )

    @property
    def _duration(self) -> Optional[timedelta]:
        return (
            pd.to_timedelta(self.duration).to_pytimedelta() if self.duration else None
        )

    def to_fv3config(self) -> Any:
        """Translate into a fv3config dictionary"""
        # overlays (i.e. diffs) requiring only a small number of inputs. In
        # particular, overlays should not require access to the full configuration
        # dictionary.
        config = fv3kube.merge_fv3config_overlays(
            fv3kube.get_base_fv3config(self.base_version),
            self._initial_condition_overlay(),
            SUPPRESS_RANGE_WARNINGS,
            dataclasses.asdict(self.to_runtime_config()),
            self._to_fv3config_specific().asdict(),
            _diag_table_overlay(self.fortran_diagnostics),
            file_configs_to_namelist_settings(
                self.fortran_diagnostics, self._physics_timestep()
            ),
            {"zhao_carr_emulation": self.zhao_carr_emulation.to_dict()},
        )
        return (
            fv3config.set_run_duration(config, self._duration)
            if self._duration
            else config
        )


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "user_config",
        type=str,
        help="Path to a config update YAML file specifying the changes from the base"
        "fv3config (e.g. diag_table, runtime, ...) for the prognostic run.",
    )
    return parser


def _diag_table_overlay(
    fortran_diagnostics: Sequence[FortranFileConfig],
    name: str = "prognostic_run",
    base_time: datetime = datetime(2000, 1, 1),
) -> Mapping[str, fv3config.DiagTable]:
    file_configs = [
        fortran_diagnostic.to_fv3config_diag_file_config()
        for fortran_diagnostic in fortran_diagnostics
    ]
    diag_table = fv3config.DiagTable(name, base_time, file_configs)
    return {"diag_table": diag_table}


def to_fv3config(dict_: dict,) -> dict:
    """Convert a loaded prognostic run yaml ``dict_`` into an fv3config
    dictionary depending on some options

    See the arguments for setting initial conditions and other runtime model
    configurations

    Args:
        ``dict_``:  a dictionary containing prognostic run configurations.  This
            dictionary combines fv3config-related keys with
            :py:class:`runtime.config.UserConfig` settings.

    Returns:
        an fv3config configuration dictionary that can be operated on with
        fv3config APIs.
    """
    user_config = HighLevelConfig.from_dict(dict_)

    if user_config.nudging and user_config.scikit_learn:
        raise NotImplementedError(
            "Nudging and machine learning cannot currently be run at the same time."
        )

    return user_config.to_fv3config()


def prepare_config(args):
    # Get model config with prognostic run updates
    with open(args.user_config, "r") as f:
        dict_ = yaml.safe_load(f)

    final = to_fv3config(dict_)
    fv3config.dump(final, sys.stdout)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()
    prepare_config(args)
