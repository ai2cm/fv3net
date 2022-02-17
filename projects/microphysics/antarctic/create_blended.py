import argparse
import json
import logging
import numpy as np
import os
import tensorflow as tf
from typing import Mapping

from fv3fit._shared import put_dir
from fv3fit.keras.adapters import get_inputs
from fv3fit.emulation.compositions import blended_model


logger = logging.getLogger(__name__)


def get_model_inputs(model1, model2):

    in1 = get_inputs(model1)
    in2 = get_inputs(model2)

    overlap = set(in1) & set(in2)

    for key in overlap:
        assert in1[key].shape.as_list() == in2[key].shape.as_list()

    combined = dict(**in1)
    combined.update(in2)

    return combined


def get_antarctic_mask(data: Mapping[str, tf.Tensor]) -> tf.Tensor:

    return data["latitude"] < np.deg2rad(-60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("antarctic_model", type=str)
    parser.add_argument("default_model", type=str)
    parser.add_argument("out_url", type=str)

    args = parser.parse_args()

    antarctic = tf.keras.models.load_model(args.antarctic_model)
    logger.info(f"Loaded antarctic model from: {args.antarctic_model}")
    default = tf.keras.models.load_model(args.default_model)
    logger.info(f"Loaded default model from: {args.default_model}")

    inputs = get_model_inputs(antarctic, default)
    inputs["latitude"] = tf.keras.Input(1, name="latitude")

    blended = blended_model(antarctic, default, get_antarctic_mask, inputs)

    with put_dir(args.out_url) as tmpdir:
        blended.compile()
        blended.save(os.path.join(tmpdir, "model.tf"))
        logger.info(f"Saving new blended model to: {args.out_url} ")

        with open(os.path.join(tmpdir, "model_info.json"), "w") as f:
            json.dump(
                {
                    "antarctic_model": args.antarctic_model,
                    "default_model": args.default_model,
                },
                f,
                indent=0
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
