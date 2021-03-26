import numpy as np
import fv3fit
import xarray as xr
import tensorflow as tf
import argparse
import random
import vcm
import yaml
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrays_dir", type=str, help="directory containing TrainingArrays data"
    )
    parser.add_argument(
        "config", type=str, help="yaml file with training configuration"
    )
    parser.add_argument(
        "model_output_dir", type=str, help="directory to write the trained model"
    )
    return parser


def load_dataset(path):
    ds = xr.open_dataset(path)
    ds.load()
    return ds


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

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    regularizer = getattr(tf.keras.regularizers, config["regularizer"]["name"])(**config["regularizer"]["kwargs"])
    optimizer_class = getattr(tf.keras.optimizers, config["optimizer"]["name"])
    optimizer_kwargs = config["optimizer"]["kwargs"]
    optimizer = optimizer_class(**optimizer_kwargs)
    input_names = config["input_variables"]


    with open(first_filename, "rb") as f:
        ds = xr.open_dataset(f)
        sample_dim_name = ds["air_temperature"].dims[0]
        model = fv3fit.keras.BPTTModel(
            sample_dim_name,
            input_names,
            kernel_regularizer=regularizer,
            optimizer=optimizer,
            **config["hyperparameters"],
        )
        model.build_for(ds)

    training_datasets = vcm.Preloaded(filenames[:-3], loader_function=load_dataset)
    validation = xr.open_dataset(filenames[-1])

    base_epoch = 0
    for i_epoch in range(config["total_epochs"]):
        epoch = base_epoch + i_epoch
        print(f"starting epoch {epoch}")
        if i_epoch == 40:
            optimizer_kwargs["lr"] = config["decreased_learning_rate"]
            model.train_keras_model.compile(
                optimizer=optimizer_class(**optimizer_kwargs),
                loss=model.losses
            )
        for i, ds in enumerate(training_datasets):
            model.fit(ds, epochs=1)
        val_loss = model.loss(validation)
        print(f"val_loss: {val_loss}")
        dirname = os.path.join(args.model_output_dir, f"model-epoch_{epoch:03d}-loss_{val_loss:.04f}")
        os.makedirs(dirname, exist_ok=True)
        fv3fit.dump(model.predictor_model, dirname)
