import dataclasses
import argparse
import yaml
import logging
import sys
from datetime import datetime, timedelta
from typing import Mapping, Sequence, Optional

import dacite

import fv3config
import fv3kube

from runtime.diagnostics.manager import FortranFileConfig
from runtime.diagnostics.fortran import file_configs_to_namelist_settings
from runtime.config import UserConfig
from runtime.steppers.machine_learning import MachineLearningConfig


__all__ = ["to_fv3config", "InitialCondition"]


logger = logging.getLogger(__name__)

PROGNOSTIC_DIAG_TABLE = "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"
SUPPRESS_RANGE_WARNINGS = {"namelist": {"fv_core_nml": {"range_warn": False}}}
FV3CONFIG_KEYS = {
    "namelist",
    "experiment_name",
    "diag_table",
    "data_table",
    "field_table",
    "initial_conditions",
    "forcing",
    "orographic_forcing",
    "patch_files",
    "gfs_analysis_data",
}


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
) -> UserConfig:
    """Ideally this function could be replaced by dacite.from_dict
    without needing any information from args.
    """
    if "nudging" in config_dict:
        config_dict["nudging"]["restarts_path"] = config_dict["nudging"].get(
            "restarts_path", nudging_url
        )

    user_config = dacite.from_dict(UserConfig, config_dict)

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


FV3Config = dict


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


def get_physics_timestep(dict_):
    base_version = dict_["base_version"]
    fv3_config = {key: dict_[key] for key in FV3CONFIG_KEYS if key in dict_}
    return timedelta(
        seconds=fv3kube.merge_fv3config_overlays(
            fv3kube.get_base_fv3config(base_version), fv3_config
        )["namelist"]["coupler_nml"]["dt_atmos"]
    )


def to_fv3config(
    dict_: dict,
    nudging_url: str,
    model_url: Sequence[str] = (),
    initial_condition: Optional[InitialCondition] = None,
    diagnostic_ml: bool = False,
) -> FV3Config:
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
    user_config = user_config_from_dict_and_args(
        dict_,
        nudging_url=nudging_url,
        model_url=model_url,
        diagnostic_ml=diagnostic_ml,
    )

    # overlays (i.e. diffs) requiring only a small number of inputs. In
    # particular, overlays should not require access to the full configuration
    # dictionary.
    return fv3kube.merge_fv3config_overlays(
        fv3kube.get_base_fv3config(dict_["base_version"]),
        {} if initial_condition is None else initial_condition.overlay,
        SUPPRESS_RANGE_WARNINGS,
        dataclasses.asdict(user_config),
        {key: dict_[key] for key in FV3CONFIG_KEYS if key in dict_},
        _diag_table_overlay(user_config.fortran_diagnostics),
        file_configs_to_namelist_settings(
            user_config.fortran_diagnostics, get_physics_timestep(dict_)
        ),
    )


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
