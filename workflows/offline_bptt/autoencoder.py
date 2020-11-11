from train import Block, WindowedDataIterator, load_window
from block_config import n_blocks_train, n_blocks_val
import fv3fit
import os
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from glob import glob
import random


def build_dense_model(
    n_state, units, n_encoder_layers, y_scaler, dropout=0.0,
):
    state_input = tf.keras.layers.Input(shape=[n_state])
    x = y_scaler.normalize_layer(state_input)
    for i in range(n_encoder_layers):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_encoder_{i}")(
            x
        )
        x = tf.keras.layers.Dropout(dropout)(x)
    encoder_output = x
    for i in range(n_encoder_layers):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_decoder_{i}")(
            x
        )
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_state, activation=None)(x)
    state_output = y_scaler.denormalize_layer(x)
    model = tf.keras.Model(inputs=state_input, outputs=state_output)
    encoder_model = tf.keras.Model(inputs=state_input, outputs=encoder_output)
    return model, encoder_model


def build_convolutional_model(
    n_state, units, n_encoder_layers, y_scaler, dropout=0.0,
):
    state_input = tf.keras.layers.Input(shape=[n_state])
    x = y_scaler.normalize_layer(state_input)
    x = tf.keras.layers.Reshape([2, n_state // 2])(x)  # switch to [variable, z] indexes
    # convolutional networks require "channels" to be last
    # so we move variable name to last index
    x = tf.keras.layers.Permute((2, 1))(x)
    # padding must happen with z at axis 1
    x = tf.keras.layers.ZeroPadding1D((0, 1))(
        x
    )  # pad z-axis to len 80 which is more divisible by 2

    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_encoder_1_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_encoder_1_2"
    )(x)
    x = tf.keras.layers.AveragePooling1D(2, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_encoder_2_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_encoder_2_2"
    )(x)
    x = tf.keras.layers.AveragePooling1D(2, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_encoder_3_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_encoder_3_2"
    )(x)
    x = tf.keras.layers.AveragePooling1D(2, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_encoder_4_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_encoder_4_2"
    )(x)
    x = tf.keras.layers.AveragePooling1D(2, padding="same")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    encoder_output = x

    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_decoder_1_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_decoder_1_2"
    )(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_decoder_2_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        8, 3, activation="relu", padding="same", name=f"conv_decoder_2_2"
    )(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_decoder_3_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_decoder_3_2"
    )(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_decoder_4_1"
    )(x)
    x = tf.keras.layers.Conv1D(
        4, 3, activation="relu", padding="same", name=f"conv_decoder_4_2"
    )(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        2, 3, activation=None, padding="same", name=f"conv_decoder_final"
    )(x)
    # cropping must happen with z at axis 1
    x = tf.keras.layers.Cropping1D((0, 1))(x)
    x = tf.keras.layers.Permute((2, 1))(x)
    x = tf.keras.layers.Reshape([n_state])(x)
    state_output = y_scaler.denormalize_layer(x)
    model = tf.keras.Model(inputs=state_input, outputs=state_output)
    model.summary()
    encoder_model = tf.keras.Model(inputs=state_input, outputs=encoder_output)
    return model, encoder_model


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    tf.random.set_seed(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("label", action="store", type=str)
    parser.add_argument("--cache-dir", action="store", type=str, default=".")
    args = parser.parse_args()
    cache_filename_y_packer = os.path.join(args.cache_dir, f"y_packer-{args.label}.yml")
    cache_filename_y_scaler = os.path.join(args.cache_dir, f"y_scaler-{args.label}.npz")

    data_fraction = 0.125  # fraction of data to use from a window
    n_blocks_window = 5
    n_blocks_between_window = n_blocks_window

    units = 32
    n_encoder_layers = 2
    dropout = 0.5
    noise_scale = 1.0
    batch_size = 48 * 12

    block_dir = f"block-{args.label}"
    autoencoder_dir = f"autoencoders-{args.label}"
    block_filenames = sorted(
        [os.path.join(block_dir, filename) for filename in os.listdir(block_dir)]
    )
    train_block_filenames = block_filenames[:n_blocks_train]
    val_block_filenames = block_filenames[
        n_blocks_train : n_blocks_train + n_blocks_val
    ]

    with open(train_block_filenames[0], "rb") as f:
        block = Block.load(f)
        y_ref = block.y_ref

    if os.path.isfile(cache_filename_y_scaler):
        with open(cache_filename_y_scaler, "rb") as f:
            y_scaler = fv3fit.keras._models.normalizer.LayerStandardScaler.load(f)
    else:
        y_scaler = fv3fit.keras._models.normalizer.LayerStandardScaler()
        y_scaler.fit(y_ref)
        with open(cache_filename_y_scaler, "wb") as f:
            y_scaler.dump(f)

    with open(cache_filename_y_packer, "r") as f:
        y_packer = fv3fit.ArrayPacker.load(f)

    batch_idx = np.random.choice(
        np.arange(y_ref.shape[0]),
        size=int(0.125 * 0.125 * y_ref.shape[0]),
        replace=False,
    )
    _, y_coarse_val, _, y_ref_val = load_window(batch_idx, val_block_filenames)
    val_shape = [y_coarse_val.shape[0] * y_coarse_val.shape[1], y_coarse_val.shape[2]]
    y_val = np.concatenate(
        [y_coarse_val.reshape(val_shape), y_ref_val.reshape(val_shape)], axis=0
    )
    y_noise = np.random.randn(*y_val.shape) * noise_scale
    y_noise *= y_scaler.std[None, :]
    y_val_noisy = y_val + y_noise

    optimizer = tf.keras.optimizers.Adam(lr=0.0001, clipnorm=1.0)

    loss = fv3fit.keras._models.loss.get_weighted_mse(
        y_packer, y_scaler.std, air_temperature=79, specific_humidity=79,
    )

    if not os.path.isdir(autoencoder_dir):
        os.mkdir(autoencoder_dir)
    model_filenames = glob(os.path.join(autoencoder_dir, "autoencoder-*.tf"))
    if len(model_filenames) > 0:
        last_model_filename = sorted(model_filenames, key=lambda x: x[-6:-3])[-1]
        model = tf.keras.models.load_model(
            last_model_filename, custom_objects={"custom_loss": loss},
        )
        base_epoch = int(last_model_filename[-6:-3]) + 1
        print(f"loaded model, resuming at epoch {base_epoch}")
    else:
        model, _ = build_dense_model(
            y_ref.shape[-1],
            units=units,
            n_encoder_layers=n_encoder_layers,
            y_scaler=y_scaler,
            dropout=dropout,
        )
        base_epoch = 0

    model.compile(
        optimizer=optimizer, loss=loss,
    )

    X_train = WindowedDataIterator(
        train_block_filenames,
        n_blocks_window=n_blocks_window,
        n_blocks_between_window=n_blocks_between_window,
        batch_size=int(48 * 48 * 6 * data_fraction),
    )

    val_loss = loss(y_val, model.predict([y_val_noisy], batch_size=batch_size),)
    print(f"validation loss {val_loss}")
    input_loss = loss(y_val, y_val_noisy,)
    print(f"input loss {input_loss}")

    for i_epoch in range(50):
        epoch = base_epoch + i_epoch
        try:
            print(f"starting epoch {epoch}")
            for _, y_coarse, _, y_ref in X_train:
                print(y_val.shape)
                # collapse windows into samples
                train_shape = [y_coarse.shape[0] * y_coarse.shape[1], y_coarse.shape[2]]
                y = np.concatenate(
                    [y_coarse.reshape(train_shape), y_ref.reshape(train_shape)], axis=0
                )
                del y_coarse
                del y_ref

                y_noise = np.random.randn(*y.shape) * noise_scale
                y_noise *= y_scaler.std[None, :]

                model.fit(
                    x=y + y_noise,
                    y=y,
                    batch_size=batch_size,
                    validation_data=(y_val_noisy, y_val),
                    epochs=1,
                    shuffle=True,
                )
                del y
            val_loss = loss(y_val, model.predict([y_val_noisy], batch_size=batch_size),)
            print(f"validation loss {val_loss}")
            print(f"saving model for epoch {epoch}")
            model.save(
                os.path.join(
                    autoencoder_dir,
                    f"autoencoder-{args.label}-{val_loss:.2f}-{epoch:03d}.tf",
                )
            )
        except KeyboardInterrupt:
            break

    y_pred = model.predict(y_val)
    plt.figure()
    plt.plot(y_val[30000, :], label="truth")
    plt.plot(y_val_noisy[30000, :], label="input")
    plt.plot(y_pred[30000, :], label="predicted")
    plt.legend(loc="best")
    plt.show()
