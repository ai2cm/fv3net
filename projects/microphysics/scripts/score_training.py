import argparse
from dataclasses import asdict
from toolz import get
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from fv3net.artifacts.metadata import StepMetadata, log_fact_json
import numpy as np
import os
import tensorflow as tf

from fv3fit import set_random_seed
from fv3fit.train_microphysics import TrainConfig
from fv3fit._shared import put_dir
from fv3fit.emulation.scoring import score_multi_output, ScoringOutput
from fv3fit.emulation.types import TensorDict
from fv3fit.wandb import (
    log_profile_plots,
    log_to_table,
)
from vcm import get_fs
import yaml
from emulation.config import EmulationConfig, ModelConfig
from emulation.masks import Mask

logger = logging.getLogger(__name__)


ArrayDict = Mapping[str, np.ndarray]
ArrayLikeDict = Union[ArrayDict, TensorDict]


def load_final_model_or_checkpoint(train_out_url) -> Tuple[tf.keras.Model, str]:

    model_url = os.path.join(train_out_url, "model.tf")
    checkpoints = os.path.join(train_out_url, "checkpoints", "*.tf")

    fs = get_fs(train_out_url)
    if fs.exists(model_url):
        logger.info(f"Loading model for scoring from: {model_url}")
        url_to_load = model_url
    elif fs.glob(checkpoints):
        url_to_load = sorted(fs.glob(checkpoints))[-1]
        logger.info(f"Loading last model checkpoint for scoring from: {url_to_load}")
    else:
        raise FileNotFoundError(f"No keras models found at {train_out_url}")

    return tf.keras.models.load_model(url_to_load), url_to_load


def _transpose(d):
    return {k: np.array(v).T for k, v in d.items()}


def _apply_mask_with_transpose(mask, emulated, state):
    # prognostic run data is [feature, sample] dimensions
    out = mask(_transpose(state), _transpose(emulated))
    return _transpose(out)


def score_model(
    targets: ArrayLikeDict, predictions: ArrayLikeDict, mask: Optional[Mask] = None
) -> ScoringOutput:

    if mask is not None:
        predictions = _apply_mask_with_transpose(mask, predictions, targets)

    names = sorted(set(predictions) & set(targets))
    return score_multi_output(get(names, targets), get(names, predictions), names)


def main(
    config: TrainConfig,
    seed: int = 0,
    model_url: Optional[str] = None,
    prognostic_emu_config: Optional[ModelConfig] = None,
    out_url_override: Optional[str] = None,
):

    logging.basicConfig(level=getattr(logging, config.log_level))
    set_random_seed(seed)

    if config.use_wandb:
        d = asdict(config)
        d["model_url_override"] = model_url
        config.wandb.init(config=d)

    if model_url is None:
        model, model_url = load_final_model_or_checkpoint(config.out_url)
    else:
        logger.info(f"Loading user specified model from {model_url}")
        model = tf.keras.models.load_model(model_url)

    if prognostic_emu_config:
        mask = prognostic_emu_config.build_mask()
    else:
        mask = None

    out_url = config.out_url if out_url_override is None else out_url_override

    StepMetadata(
        job_type="train_score",
        url=out_url,
        dependencies=dict(
            train_data=config.train_url, test_data=config.test_url, model=model_url
        ),
        args=sys.argv[1:],
    ).print_json()

    train_ds = config.open_dataset(
        config.train_url, config.nfiles, config.model_variables
    )
    test_ds = config.open_dataset(
        config.train_url, config.nfiles, config.model_variables
    )

    train_set = next(iter(train_ds.unbatch().shuffle(160_000).batch(80_000)))
    test_set = next(iter(test_ds.unbatch().shuffle(160_000).batch(80_000)))
    train_predictions = model.predict(train_set, batch_size=1024)
    test_predictions = model.predict(test_set, batch_size=1024)

    train_scores, train_profiles = score_model(train_set, train_predictions, mask=mask)
    test_scores, test_profiles = score_model(test_set, test_predictions, mask=mask)
    logger.debug("Scoring Complete")

    summary_metrics: Dict[str, Any] = {
        f"score/train/{key}": value for key, value in train_scores.items()
    }
    summary_metrics.update(
        {f"score/test/{key}": value for key, value in test_scores.items()}
    )

    # Logging for google cloud
    log_fact_json(data=summary_metrics)

    if config.use_wandb:
        pred_sample = model.predict(test_set, batch_size=8192)
        log_profile_plots(test_set, pred_sample)

        # add level for dataframe index, assumes equivalent feature dims
        sample_profile = next(iter(train_profiles.values()))
        train_profiles["level"] = np.arange(len(sample_profile))
        test_profiles["level"] = np.arange(len(sample_profile))

        config.wandb.job.log(summary_metrics)

        log_to_table("profiles/train", train_profiles)
        log_to_table("profiles/test", test_profiles)

    with put_dir(out_url) as tmpdir:
        with open(os.path.join(tmpdir, "scores.json"), "w") as f:
            json.dump({"train": train_scores, "test": test_scores}, f)


def _load_prognostic_emulator_model_config(config: Path) -> ModelConfig:
    with config.open() as f:
        d = yaml.safe_load(f)

    emu_config = EmulationConfig.from_dict(d)
    model_config = emu_config.get_defined_model_config()
    return model_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_url",
        help=(
            "Specify model path to run scoring for. Overrides use of models "
            "at the config.out_url"
        ),
        default=None,
    )
    parser.add_argument(
        "--emulator_config",
        type=Path,
        required=False,
        help=(
            "Use EmulatorConfig for post-hoc emulation corrections. "
            "Overrides TrainConfig out_url."
        ),
    )
    parser.add_argument(
        "--out_url",
        type=str,
        required=False,
        help="Override the save location for the scoring metrics",
    )

    known, train_config_args = parser.parse_known_args()

    if known.emulator_config is not None:
        model_config = _load_prognostic_emulator_model_config(known.emulator_config)
        model_url = model_config.path
    else:
        model_url = known.model_url
        model_config = None

    train_config = TrainConfig.from_args(train_config_args)

    main(
        train_config,
        model_url=model_url,
        prognostic_emu_config=model_config,
        out_url_override=known.out_url,
    )
