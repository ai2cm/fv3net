import argparse
from dataclasses import asdict
from toolz import get
import logging
from pathlib import Path
import sys
from typing import MutableMapping, Mapping, Optional, Tuple
from fv3net.artifacts.metadata import StepMetadata, log_fact_json
import numpy as np
import os
import tensorflow as tf

from fv3fit import set_random_seed
from fv3fit.train_microphysics import TrainConfig
from fv3fit.emulation.scoring import score_multi_output, ScoringOutput
from fv3fit.emulation.transforms.zhao_carr import CLASS_NAMES
from fv3fit.wandb import (
    log_profile_plots,
    log_to_table,
)
import vcm
from vcm import get_fs
import yaml
from emulation.config import EmulationConfig
from emulation.masks import Mask
from emulation.zhao_carr import Input, GscondOutput

logger = logging.getLogger(__name__)

REQUIRED_VARS = set(
    [Input.cloud_water, Input.humidity, GscondOutput.cloud_water, GscondOutput.humidity]
)


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
    # prognostic runtime mask expects [feature, sample] dimensions
    out = mask(_transpose(state), _transpose(emulated))
    return _transpose(out)


def score_model(
    targets: Mapping, predictions: Mapping, mask: Optional[Mask] = None
) -> ScoringOutput:

    targets: MutableMapping = {**targets}
    if mask is not None:
        predictions = _apply_mask_with_transpose(mask, predictions, targets)

    names = sorted(set(predictions) & set(targets))
    return score_multi_output(get(names, targets), get(names, predictions), names)


def logit_to_one_hot(x):
    return x == tf.math.reduce_max(x, axis=-1, keepdims=True)


def score_gscond_classes(targets: Mapping, predictions: Mapping):
    one_hot_encoded_name = "gscond_classes"
    if one_hot_encoded_name not in predictions:
        return {}, {}

    y = predictions[one_hot_encoded_name]
    # expected shape of y is [..., height, number of classes]
    predicted_class = logit_to_one_hot(y).numpy()
    truth = targets[one_hot_encoded_name].numpy()

    profiles = {}
    scalars = {}

    for score in [vcm.accuracy, vcm.f1_score, vcm.precision, vcm.recall]:
        # average over final dimension: number of classes
        profile = score(truth, predicted_class, mean=lambda x: x.mean(0))
        # average over height and classes
        integral = score(truth, predicted_class, mean=lambda x: x.mean(0).mean(0))

        score_name = score.__name__

        names = sorted(CLASS_NAMES)
        for i, class_name in enumerate(names):
            profiles[f"{score_name}/{class_name}"] = profile[..., i]
            scalars[f"{score_name}/{class_name}"] = integral[..., i].item()

    scalars["accuracy/all"] = vcm.accuracy(truth, predicted_class, lambda x: x.mean())
    profiles["accuracy/all"] = vcm.accuracy(
        truth, predicted_class, lambda x: x.mean(0).mean(-1)
    )

    return scalars, profiles


def main(
    config: TrainConfig,
    seed: int = 0,
    model_url: Optional[str] = None,
    emulation_mask: Optional[Mask] = None,
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

    StepMetadata(
        job_type="train_score",
        url=config.out_url,
        dependencies=dict(
            train_data=config.train_url, test_data=config.test_url, model=model_url
        ),
        args=sys.argv[1:],
    ).print_json()

    vars_to_include = config.model_variables | REQUIRED_VARS
    train_ds = config.open_dataset(config.train_url, config.nfiles, vars_to_include)
    test_ds = config.open_dataset(config.test_url, config.nfiles, vars_to_include)
    n = 80_000
    train_set = next(iter(train_ds.unbatch().shuffle(2 * n).batch(n)))
    test_set = next(iter(test_ds.unbatch().shuffle(2 * n).batch(n)))

    summary_metrics = {}

    for split_name, data in [("train", train_set), ("test", test_set)]:
        transform = config.build_transform(train_set)
        targets = transform.backward(transform.forward(data))
        predictions = model.predict(data, batch_size=8192)

        scores, profiles = score_model(targets, predictions, mask=emulation_mask)

        summary_metrics.update(
            {f"score/{split_name}/{key}": value for key, value in scores.items()}
        )
        class_scores, class_profiles = score_gscond_classes(targets, predictions)
        scores.update(class_scores)

        for score, value in scores.items():
            summary_metrics[f"score/{split_name}/{score}"] = value

        all_profiles = {**profiles, **class_profiles}
        # add level for dataframe index, assumes equivalent feature dims
        sample_profile = next(iter(all_profiles.values()))
        all_profiles["level"] = np.arange(len(sample_profile))

        if config.use_wandb:
            log_to_table(f"profiles/{split_name}", all_profiles)

    logger.debug("Scoring Complete")

    # log scalar metrics
    # Logging for google cloud
    log_fact_json(data=summary_metrics)
    if config.use_wandb:
        config.wandb.job.log(summary_metrics)

    # log individual profile predictions
    if config.use_wandb:
        pred_sample = model.predict(test_set, batch_size=8192)
        log_profile_plots(test_set, pred_sample)


def _get_defined_model_config(emu_config: EmulationConfig):
    # Based on GFS_physics_driver only one of model or gscond
    # should be provided, return whichever that is

    if emu_config.model is None and emu_config.gscond is None:
        raise ValueError(
            "Both model and gscond attributes are undefind for provided EmulationConfig"
        )

    if emu_config.model is not None:
        return emu_config.model
    else:
        return emu_config.gscond


def get_mask_and_emu_url_from_prog_config(
    config: Optional[Path] = None,
) -> Tuple[Optional[Mask], Optional[str]]:
    if config is None:
        return None, None

    with config.open() as f:
        d = yaml.safe_load(f) or {}

    if "zhao_carr_emulation" in d:

        emu_config = EmulationConfig.from_dict(d["zhao_carr_emulation"])
        model_config = emu_config.get_defined_model_config()

        model_url = model_config.path
        mask = model_config.build_mask()
        return mask, model_url
    else:
        return None, None


def get_train_config_from_model_url(model_url: str) -> str:
    train_config_parent = "/".join(model_url.split("/")[:-1])
    train_config_url = os.path.join(train_config_parent, "config.yaml")

    return train_config_url


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-url",
        help=(
            "Specify model path to run scoring for. Overrides use of models "
            "in the training configuration or prognostic emulator config."
        ),
        default=None,
    )
    parser.add_argument(
        "--prognostic-config",
        type=Path,
        required=False,
        help=(
            "Use EmulatorConfig for post-hoc emulation adjustments when scoring. "
            "Overrides training configuration model."
        ),
    )

    known, train_config_args = parser.parse_known_args()

    mask, prognostic_emu_model_url = get_mask_and_emu_url_from_prog_config(
        known.prognostic_config
    )

    # preference to model_url override if specified
    model_url = known.model_url or prognostic_emu_model_url
    if model_url is not None:
        train_config_url = get_train_config_from_model_url(model_url)
        # argparse uses last found match, so overrides if already present in args
        train_config_args.extend(["--config-path", train_config_url])

    config = TrainConfig.from_args(train_config_args)
    main(
        config, model_url=model_url, emulation_mask=mask,
    )
