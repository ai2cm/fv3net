from typing import Iterable, Hashable, Sequence, Tuple
import fv3gfs.util
import numpy as np
import fv3fit
import xarray as xr
import tensorflow as tf
import argparse
import concurrent.futures
from download import INPUT_NAMES
import copy
import random
import vcm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrays_dir", type=str, help="directory containing TrainingArrays data"
    )
    parser.add_argument(
        "model_output_dir", type=str, help="directory to write the trained model"
    )
    return parser


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    tf.random.set_seed(2)
    # tf.config.experimental.enable_mlir_graph_optimization()
    parser = get_parser()
    args = parser.parse_args()

    fs = vcm.get_fs(args.arrays_dir)
    filenames = sorted(fs.listdir(args.arrays_dir, detail=False))
    first_filename = filenames[0]

    with open(first_filename, "rb") as f:
        ds = xr.open_dataset(f)
        sample_dim_name = ds["air_temperature"].dims[0]
        model = fv3fit.keras.BPTTModel(
            sample_dim_name,
            INPUT_NAMES,
            n_units=32,
            n_hidden_layers=3,
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            train_batch_size=48,
            optimizer=tf.keras.optimizers.Adam(lr=0.001, amsgrad=True, clipnorm=1.0),
        )
        model.build_for(ds)

    training_datasets = PreloadedIterator(filenames, loader_function=xr.open_dataset)

    base_epoch = 0
    for i_epoch in range(50):
        epoch = base_epoch + i_epoch
        print(f"starting epoch {epoch}")
        for i, ds in enumerate(training_datasets):
            model.fit(ds, epochs=1)

    fv3fit.dump(model.predictor_model, args.model_output_dir)
