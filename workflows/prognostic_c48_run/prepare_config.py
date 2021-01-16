import dataclasses
import argparse
import yaml
import logging
from typing import Dict, List, Optional, Sequence, Mapping

import fv3config
import fv3kube

import vcm

from runtime import default_diagnostics
import runtime.diagnostics.manager
from runtime.diagnostics.manager import DiagnosticFileConfig, TimeConfig


@dataclasses.dataclass
class MachineLearningConfig:
    model: Sequence[str] = dataclasses.field(default_factory=list)
    diagnostic_ml: bool = False


@dataclasses.dataclass
class NudgingConfig:
    timescale_hours: Dict[str, float]
    restarts_path: str


@dataclasses.dataclass
class UserConfig:
    diagnostics: List[runtime.diagnostics.manager.DiagnosticFileConfig]
    scikit_learn: MachineLearningConfig = MachineLearningConfig()
    nudging: Optional[NudgingConfig] = None
    namelist: Mapping = dataclasses.field(default_factory=dict)
    base_version: str = "v0.5"
    step_tendency_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(
            ("specific_humidity", "air_temperature", "eastward_wind", "northward_wind",)
        )
    )
    step_storage_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(("specific_humidity", "total_water"))
    )

    @staticmethod
    def from_dict_args(config_dict: dict, args):

        nudging = (
            NudgingConfig(
                timescale_hours=config_dict["nudging"]["timescale_hours"],
                restarts_path=config_dict["nudging"].get(
                    "restarts_path", args.initial_condition_url
                ),
            )
            if "nudging" in config_dict
            else None
        )

        diagnostics = [
            runtime.diagnostics.manager.DiagnosticFileConfig.from_dict(
                diag, args.ic_timestep
            )
            for diag in config_dict.get("diagnostics", [])
        ]

        scikit_learn = MachineLearningConfig(
            model=list(args.model_url or []), diagnostic_ml=args.diagnostic_ml
        )

        default = UserConfig(diagnostics=[])

        if nudging and len(scikit_learn.model):
            raise NotImplementedError(
                "Nudging and machine learning cannot "
                "currently be run at the same time."
            )

        return UserConfig(
            nudging=nudging,
            diagnostics=diagnostics,
            namelist=config_dict.get("namelist", {}),
            scikit_learn=scikit_learn,
            base_version=config_dict["base_version"],
            step_storage_variables=config_dict.get(
                "step_storage_variables", default.step_storage_variables
            ),
            step_tendency_variables=config_dict.get(
                "step_tendency_variables", default.step_tendency_variables
            ),
        )


logger = logging.getLogger(__name__)

PROGNOSTIC_DIAG_TABLE = "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"
SUPPRESS_RANGE_WARNINGS = {"namelist": {"fv_core_nml": {"range_warn": False}}}


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
        "--nudge-to-observations", action="store_true", help="Nudge to observations",
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


def diagnostics_overlay(
    config: UserConfig,
    model_urls: List[str],
    nudge_to_obs: bool,
    frequency_minutes: int,
):

    diagnostic_files: List[DiagnosticFileConfig] = []

    if config.scikit_learn.model:
        diagnostic_files.append(default_diagnostics.ml_diagnostics)
    elif config.nudging or nudge_to_obs:
        diagnostic_files.append(default_diagnostics.state_after_timestep)
        diagnostic_files.append(default_diagnostics.physics_tendencies)
        if config.nudging:
            diagnostic_files.append(_nudging_tendencies(config.nudging))
            diagnostic_files.append(default_diagnostics.nudging_diagnostics_2d)
            diagnostic_files.append(_reference_state(config.nudging))
    else:
        diagnostic_files.append(default_diagnostics.baseline_diagnostics)

    _update_times(diagnostic_files, frequency_minutes)

    return {
        "diagnostics": [diag_file.to_dict() for diag_file in diagnostic_files],
        "diag_table": PROGNOSTIC_DIAG_TABLE,
    }


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


def prepare_config(args):
    # Get model config with prognostic run updates
    with open(args.user_config, "r") as f:
        user_config = yaml.safe_load(f)

    config = UserConfig.from_dict_args(user_config, args)
    _prepare_config_from_parsed_config(config, args)


def get_run_duration(config: UserConfig):
    return fv3config.get_run_duration({"namelist": config.namelist})


def _prepare_config_from_parsed_config(config: UserConfig, args):
    model_urls = args.model_url if args.model_url else []

    # To simplify the configuration flow, updates should be implemented as
    # overlays (i.e. diffs) requiring only a small number of inputs. In
    # particular, overlays should not require access to the full configuration
    # dictionary.
    overlays = [
        fv3kube.get_base_fv3config(config.base_version),
        fv3kube.c48_initial_conditions_overlay(
            args.initial_condition_url, args.ic_timestep
        ),
        diagnostics_overlay(
            config, model_urls, args.nudge_to_observations, args.output_frequency,
        ),
        SUPPRESS_RANGE_WARNINGS,
        dataclasses.asdict(config),
    ]

    if args.nudge_to_observations:
        # get timing information
        duration = get_run_duration(config)
        current_date = vcm.parse_current_date_from_str(args.ic_timestep)
        overlays.append(
            fv3kube.enable_nudge_to_observations(
                duration,
                current_date,
                nudge_url="gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data",
            )
        )

    config = fv3kube.merge_fv3config_overlays(*overlays)
    print(yaml.dump(config))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()
    prepare_config(args)
