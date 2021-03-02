import dataclasses
import argparse
import yaml
import logging
from datetime import datetime
from typing import List, Mapping, Sequence

import dacite

import fv3config
import fv3kube

from runtime import default_diagnostics
from runtime.diagnostics.manager import (
    FortranFileConfig,
    DiagnosticFileConfig,
    TimeConfig,
)
from runtime.steppers.nudging import NudgingConfig
from runtime.config import UserConfig
from runtime.steppers.machine_learning import MachineLearningConfig


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
        "--output-frequency",
        type=int,
        default=15,
        help=(
            "Output frequency (in minutes) of ML/nudging diagnostics. If omitted, "
            "output will be written every 15 minutes from the initial time."
        ),
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    return parser


def user_config_from_dict_and_args(config_dict: dict, args) -> UserConfig:
    """Ideally this function could be replaced by dacite.from_dict
    without needing any information from args.
    """
    nudge_to_observations = (
        config_dict.get("namelist", {}).get("fv_core_nml", {}).get("nudge", False)
    )

    if "nudging" in config_dict:
        config_dict["nudging"]["restarts_path"] = config_dict["nudging"].get(
            "restarts_path", args.initial_condition_url
        )
        nudging = dacite.from_dict(NudgingConfig, config_dict["nudging"])
    else:
        nudging = None

    scikit_learn = MachineLearningConfig(
        model=list(args.model_url or []), diagnostic_ml=args.diagnostic_ml
    )

    if "diagnostics" in config_dict:
        diagnostics = [
            dacite.from_dict(DiagnosticFileConfig, diag)
            for diag in config_dict["diagnostics"]
        ]
    else:
        diagnostics = _default_diagnostics(
            nudging, scikit_learn, nudge_to_observations, args.output_frequency,
        )

    if "fortran_diagnostics" in config_dict:
        fortran_diagnostics = [
            dacite.from_dict(FortranFileConfig, diag)
            for diag in config_dict["fortran_diagnostics"]
        ]
    else:
        fortran_diagnostics = _default_fortran_diagnostics(nudge_to_observations)

    default = UserConfig(diagnostics=[], fortran_diagnostics=[])

    if nudging and len(scikit_learn.model):
        raise NotImplementedError(
            "Nudging and machine learning cannot currently be run at the same time."
        )

    return UserConfig(
        nudging=nudging,
        diagnostics=diagnostics,
        fortran_diagnostics=fortran_diagnostics,
        scikit_learn=scikit_learn,
        step_storage_variables=config_dict.get(
            "step_storage_variables", default.step_storage_variables
        ),
        step_tendency_variables=config_dict.get(
            "step_tendency_variables", default.step_tendency_variables
        ),
    )


def _default_diagnostics(
    nudging: NudgingConfig,
    scikit_learn: MachineLearningConfig,
    nudge_to_obs: bool,
    frequency_minutes: int,
) -> List[DiagnosticFileConfig]:
    diagnostic_files: List[DiagnosticFileConfig] = []

    if scikit_learn.model:
        diagnostic_files.append(default_diagnostics.ml_diagnostics)
    elif nudging or nudge_to_obs:
        diagnostic_files.append(default_diagnostics.state_after_timestep)
        diagnostic_files.append(default_diagnostics.physics_tendencies)
        if nudging:
            diagnostic_files.append(_nudging_tendencies(nudging))
            diagnostic_files.append(default_diagnostics.nudging_diagnostics_2d)
            diagnostic_files.append(_reference_state(nudging))
    else:
        diagnostic_files.append(default_diagnostics.baseline_diagnostics)

    _update_times(diagnostic_files, frequency_minutes)
    return diagnostic_files


def _default_fortran_diagnostics(
    nudge_to_observations: bool,
) -> List[FortranFileConfig]:
    fortran_diags = [
        default_diagnostics.sfc_dt_atmos,
        default_diagnostics.atmos_dt_atmos,
        default_diagnostics.atmos_8xdaily,
    ]
    if nudge_to_observations:
        fortran_diags.append(default_diagnostics.nudging_tendencies_fortran)
    return fortran_diags


def _nudging_tendencies(config: NudgingConfig) -> DiagnosticFileConfig:

    nudging_tendencies = default_diagnostics.nudging_tendencies
    nudging_variables = list(config.timescale_hours)
    if isinstance(nudging_tendencies.variables, list):
        nudging_tendencies.variables.extend(
            [f"{var}_tendency_due_to_nudging" for var in nudging_variables]
        )
    return nudging_tendencies


def _reference_state(config: NudgingConfig) -> DiagnosticFileConfig:
    reference_states = default_diagnostics.reference_state
    nudging_variables = list(config.timescale_hours)
    if isinstance(reference_states.variables, list):
        reference_states.variables.extend(
            [f"{var}_reference" for var in nudging_variables]
        )

    return reference_states


def _update_times(
    diagnostic_files: List[DiagnosticFileConfig], frequency_minutes: int
) -> List[DiagnosticFileConfig]:
    for diagnostic in diagnostic_files:
        diagnostic.times = TimeConfig(kind="interval", frequency=60 * frequency_minutes)
    return diagnostic_files


def _diag_table_overlay(
    fortran_diagnostics: Sequence[FortranFileConfig],
) -> Mapping[str, fv3config.DiagTable]:
    file_configs = [
        fortran_diagnostic.to_fv3config_diag_file_config()
        for fortran_diagnostic in fortran_diagnostics
    ]
    diag_table = fv3config.DiagTable(
        "prognostic_run", datetime(2000, 1, 1), file_configs=file_configs
    )
    return {"diag_table": diag_table}


def prepare_config(args):
    # Get model config with prognostic run updates
    with open(args.user_config, "r") as f:
        dict_ = yaml.safe_load(f)

    user_config = user_config_from_dict_and_args(dict_, args)

    fv3config_config = {key: dict_[key] for key in FV3CONFIG_KEYS if key in dict_}
    final = _prepare_config_from_parsed_config(
        user_config, fv3config_config, dict_["base_version"], args
    )
    # following should be handled by fv3config
    if isinstance(final["diag_table"], fv3config.DiagTable):
        final["diag_table"] = final["diag_table"].asdict()
    print(yaml.safe_dump(final))


def _prepare_config_from_parsed_config(
    user_config: UserConfig, fv3_config: dict, base_version: str, args
):
    if not set(fv3_config) <= FV3CONFIG_KEYS:
        raise ValueError(
            f"{fv3_config.keys()} contains a key that fv3config does not handle. "
            "Python runtime configurations should be specified in the user_config "
            "argument."
        )

    # To simplify the configuration flow, updates should be implemented as
    # overlays (i.e. diffs) requiring only a small number of inputs. In
    # particular, overlays should not require access to the full configuration
    # dictionary.
    overlays = [
        fv3kube.get_base_fv3config(base_version),
        fv3kube.c48_initial_conditions_overlay(
            args.initial_condition_url, args.ic_timestep
        ),
        _diag_table_overlay(user_config.fortran_diagnostics),
        SUPPRESS_RANGE_WARNINGS,
        dataclasses.asdict(user_config),
        fv3_config,
    ]

    return fv3kube.merge_fv3config_overlays(*overlays)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()
    prepare_config(args)
