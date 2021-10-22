import dataclasses
import argparse
import yaml
import logging
import sys
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence, Optional, Union, List

import dacite

import fv3config
import fv3kube

from runtime.diagnostics.manager import FortranFileConfig
from runtime.diagnostics.fortran import file_configs_to_namelist_settings
from runtime.config import UserConfig
from runtime.steppers.machine_learning import MachineLearningConfig


__all__ = ["to_fv3config", "InitialCondition", "FV3Config", "HighLevelConfig"]


logger = logging.getLogger(__name__)

PROGNOSTIC_DIAG_TABLE = "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"
SUPPRESS_RANGE_WARNINGS = {"namelist": {"fv_core_nml": {"range_warn": False}}}


def default_coupler_nml():
    return {"coupler_nml": {"dt_atmos": 900}}


@dataclasses.dataclass
class FV3Config:
    """Dataclass representation of an fv3config object"""

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
    """

    base_url: str
    timestep: str

    @property
    def overlay(self):
        return fv3kube.c48_initial_conditions_overlay(self.base_url, self.timestep)


@dataclasses.dataclass
class HighLevelConfig(UserConfig, FV3Config):
    """A high level configuration object for prognostic runs

    Combines fv3config and runtime configurations with conveniences for commonly
    used initial conditions and configurations

    Attributes:
        base_version: the default physics config
        initial_conditions: Specification for the initial conditions
        fortran_diagnostics: list of Fortran diagnostic file configurations

    See :py:class:`runtime.config.UserConfig` and :py:class:`FV3Config` for
    documentation on other allowed attributes

    """

    base_version: str = "v0.5"
    initial_conditions: Union[InitialCondition, Any] = ""
    fortran_diagnostics: List[FortranFileConfig] = dataclasses.field(
        default_factory=list
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
        return UserConfig(
            diagnostics=self.diagnostics,
            prephysics=self.prephysics,
            scikit_learn=self.scikit_learn,
            nudging=self.nudging,
            tendency_prescriber=self.tendency_prescriber,
            online_emulator=self.online_emulator,
        )

    def _physics_timestep(self) -> timedelta:
        dict_ = dataclasses.asdict(self._to_fv3config_specific())
        return timedelta(
            seconds=fv3kube.merge_fv3config_overlays(
                fv3kube.get_base_fv3config(self.base_version), dict_
            )["namelist"]["coupler_nml"]["dt_atmos"]
        )

    def to_fv3config(self) -> Any:
        """Translate into a fv3config dictionary"""
        # overlays (i.e. diffs) requiring only a small number of inputs. In
        # particular, overlays should not require access to the full configuration
        # dictionary.
        return fv3kube.merge_fv3config_overlays(
            fv3kube.get_base_fv3config(self.base_version),
            self._initial_condition_overlay(),
            SUPPRESS_RANGE_WARNINGS,
            dataclasses.asdict(self.to_runtime_config()),
            self._to_fv3config_specific().asdict(),
            _diag_table_overlay(self.fortran_diagnostics),
            file_configs_to_namelist_settings(
                self.fortran_diagnostics, self._physics_timestep()
            ),
        )


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "user_config",
        type=str,
        help="Path to a config update YAML file specifying the changes from the base"
        "fv3config (e.g. diag_table, runtime, ...) for the prognostic run.",
    )
    parser.add_argument(
        "initial_condition_url",
        type=str,
        help="Remote url to directory holding timesteps with model initial conditions.",
    )
    parser.add_argument(
        "ic_timestep",
        type=str,
        help="YYYYMMDD.HHMMSS timestamp to grab from the initial conditions url.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        action="append",
        help=(
            "Remote url to a trained ML model. If a model is omitted (and not "
            "specified in `user_config`'s `scikit-learn` `model` field either), then "
            "no ML updating will be done. Also, if an ML model is provided, no "
            "nudging will be done. Can be provided multiple times, "
            "ex. --model_url model1 --model_url model2. If multiple urls are given, "
            "they will be combined into a single model at runtime, providing the "
            "outputs are nonoverlapping."
        ),
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    return parser


def user_config_from_dict_and_args(
    config_dict: dict, nudging_url, model_url, diagnostic_ml
) -> HighLevelConfig:
    """Ideally this function could be replaced by dacite.from_dict
    without needing any information from args.
    """
    if "nudging" in config_dict:
        config_dict["nudging"]["restarts_path"] = config_dict["nudging"].get(
            "restarts_path", nudging_url
        )
    user_config = dacite.from_dict(
        HighLevelConfig, config_dict, dacite.Config(strict=True)
    )

    # insert command line option overrides
    if user_config.scikit_learn is None:
        if model_url:
            user_config.scikit_learn = MachineLearningConfig(
                model=list(model_url), diagnostic_ml=diagnostic_ml
            )
    else:
        if model_url:
            user_config.scikit_learn.model = list(model_url)
        if diagnostic_ml:
            user_config.scikit_learn.diagnostic_ml = diagnostic_ml

    if user_config.nudging and user_config.scikit_learn:
        raise NotImplementedError(
            "Nudging and machine learning cannot currently be run at the same time."
        )

    return user_config


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


def to_fv3config(
    dict_: dict,
    nudging_url: str,
    model_url: Sequence[str] = (),
    initial_condition: Optional[InitialCondition] = None,
    diagnostic_ml: bool = False,
) -> dict:
    """Convert a loaded prognostic run yaml ``dict_`` into an fv3config
    dictionary depending on some options

    See the arguments for setting initial conditions and other runtime model
    configurations

    Args:
        ``dict_``:  a dictionary containing prognostic run configurations.  This
            dictionary combines fv3config-related keys with
            :py:class:`runtime.config.UserConfig` settings.
        initial_condition: modify the initial_conditions if provided, otherwise
            leaves ``dict_`` unchanged.

    Returns:
        an fv3config configuration dictionary that can be operated on with
        fv3config APIs.
    """
    full_config = user_config_from_dict_and_args(
        dict_,
        nudging_url=nudging_url,
        model_url=model_url,
        diagnostic_ml=diagnostic_ml,
    )

    if initial_condition:
        full_config.initial_conditions = initial_condition

    return full_config.to_fv3config()


def prepare_config(args):
    # Get model config with prognostic run updates
    with open(args.user_config, "r") as f:
        dict_ = yaml.safe_load(f)

    final = to_fv3config(
        dict_,
        initial_condition=InitialCondition(
            args.initial_condition_url, args.ic_timestep
        ),
        nudging_url=args.initial_condition_url,
        model_url=args.model_url,
        diagnostic_ml=args.diagnostic_ml,
    )
    fv3config.dump(final, sys.stdout)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()
    prepare_config(args)
