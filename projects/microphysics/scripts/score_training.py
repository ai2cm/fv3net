import argparse
from dataclasses import asdict
import json
import logging
import numpy as np
import os
import tensorflow as tf

from fv3fit import set_random_seed
from fv3fit.train_microphysics import TrainConfig
from fv3fit._shared import put_dir
from fv3fit.emulation.data import nc_dir_to_tf_dataset
from fv3fit.emulation.keras import score_model
from fv3fit.wandb import (
    log_profile_plots,
    log_to_table,
)
from vcm import get_fs

logger = logging.getLogger(__name__)


def load_final_model_or_checkpoint(train_out_url) -> tf.keras.Model:

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

    return tf.keras.models.load_model(url_to_load)


def main(config: TrainConfig, seed: int = 0, model_url: str = None):

    logging.basicConfig(level=getattr(logging, config.log_level))
    set_random_seed(seed)

    if config.use_wandb:
        d = asdict(config)
        d["model_url_override"] = model_url
        config.wandb.init(config=d)

    if model_url is None:
        model = load_final_model_or_checkpoint(config.out_url)
    else:
        logger.info(f"Loading user specified model from {model_url}")
        model = tf.keras.models.load_model(model_url)

    train_ds = nc_dir_to_tf_dataset(
        config.train_url, config.transform, nfiles=config.nfiles
    )
    test_ds = nc_dir_to_tf_dataset(
        config.test_url, config.transform, nfiles=config.nfiles_valid
    )

    train_set = next(iter(train_ds.shuffle(100_000).batch(50_000)))
    test_set = next(iter(test_ds.shuffle(160_000).batch(80_000)))

    train_scores, train_profiles = score_model(model, train_set)
    test_scores, test_profiles = score_model(model, test_set)
    logger.debug("Scoring Complete")

    if config.use_wandb:
        pred_sample = model.predict(test_set)
        log_profile_plots(test_set, pred_sample)

        # add level for dataframe index, assumes equivalent feature dims
        sample_profile = next(iter(train_profiles.values()))
        train_profiles["level"] = np.arange(len(sample_profile))
        test_profiles["level"] = np.arange(len(sample_profile))

        log_to_table("score/train", train_scores, index=[config.wandb.job.name])
        log_to_table("score/test", test_scores, index=[config.wandb.job.name])
        log_to_table("profiles/train", train_profiles)
        log_to_table("profiles/test", test_profiles)

    with put_dir(config.out_url) as tmpdir:
        with open(os.path.join(tmpdir, "scores.json"), "w") as f:
            json.dump({"train": train_scores, "test": test_scores}, f)


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
