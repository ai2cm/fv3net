import argparse
from dataclasses import asdict
import logging
import sys
from typing import Tuple
from fv3net.artifacts.metadata import StepMetadata, log_fact_json
import numpy as np
import os
import tensorflow as tf

from fv3fit import set_random_seed
from fv3fit.train_microphysics import TrainConfig
from fv3fit.emulation.keras import score_model
from fv3fit.emulation.transforms.zhao_carr import CLASS_NAMES
from fv3fit.wandb import (
    log_profile_plots,
    log_to_table,
)
from vcm import get_fs
import vcm

logger = logging.getLogger(__name__)


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


def logit_to_one_hot(x):
    return x == tf.math.reduce_max(x, axis=-1, keepdims=True)


def score_gscond_classes(config, model, batch):
    v = "gscond_classes"
    if v not in model.output_names:
        return {}, {}

    transform = config.build_transform(batch)
    batch_transformed = transform.forward(batch)
    out = model(batch)

    y = out[v]
    predicted_class = logit_to_one_hot(y).numpy()
    truth = batch_transformed[v].numpy()

    profiles = {}
    scalars = {}

    for score in [vcm.accuracy, vcm.f1_score, vcm.precision, vcm.recall]:
        profile = score(truth, predicted_class, mean=lambda x: x.mean(0))
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


def main(config: TrainConfig, seed: int = 0, model_url: str = None):

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

    train_ds = config.open_dataset(
        config.train_url, config.nfiles, config.model_variables
    )
    test_ds = config.open_dataset(
        config.train_url, config.nfiles, config.model_variables
    )
    n = 80_000
    train_set = next(iter(train_ds.unbatch().shuffle(2 * n).batch(n)))
    test_set = next(iter(test_ds.unbatch().shuffle(2 * n).batch(n)))

    summary_metrics = {}
    profiles = {}

    for split_name, data in [("train", train_set), ("test", test_set)]:
        scores, profiles = score_model(model, data)
        summary_metrics.update(
            {f"score/{split_name}/{key}": value for key, value in scores.items()}
        )
        class_scores, class_profiles = score_gscond_classes(config, model, data)
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

    known, unknown = parser.parse_known_args()
    config = TrainConfig.from_args(unknown)
    main(config, model_url=known.model_url)
