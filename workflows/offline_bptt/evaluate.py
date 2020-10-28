from train import load_window
import fv3fit
import numpy as np
import random
import tensorflow as tf
import os
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(1)
    tf.random.set_seed(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("label", action="store", type=str)
    parser.add_argument("--cache-dir", action="store", type=str, default=".")
    args = parser.parse_args()
    block_dir = f"block-{args.label}"

    n_block = 12  # number of times in a npz file
    blocks_per_day = 24 // n_block
    n_blocks_train = 10 * blocks_per_day
    n_blocks_val = 5 * blocks_per_day

    all_filenames = sorted(
        [os.path.join(block_dir, filename) for filename in os.listdir(block_dir)]
    )
    val_block_filenames = all_filenames[n_blocks_train : n_blocks_train + n_blocks_val]

    model_dir = f"models-{args.label}"
    model_filenames = os.listdir(model_dir)
    if len(model_filenames) > 0:
        last_model_filename = os.path.join(
            model_dir, sorted(model_filenames, key=lambda x: x[-6:-3])[-1]
        )
        model = tf.keras.models.load_model(
            last_model_filename,
            custom_objects={
                "custom_loss": tf.keras.losses.mse,
                "GCMCell": fv3fit.keras._models.gcm_cell.GCMCell,
            },
        )
        epoch = int(last_model_filename[-6:-3])
        print(f"loaded model for epoch {epoch}")
    else:
        raise RuntimeError(f"found no models at {model_dir}")

    n_plots = 10
    idx = np.random.choice(np.arange(48 * 48 * 6), size=[n_plots])
    X_coarse_val, y_coarse_val, y_ref_val = load_window(idx, val_block_filenames)
    y_pred_val = model.predict([X_coarse_val, y_coarse_val])
    for i in range(n_plots):

        z_max = 79
        coarse = y_coarse_val[i, :, 79 + z_max - 1 : 79 - 1 : -1].T
        ref = y_ref_val[i, :, 79 + z_max - 1 : 79 - 1 : -1].T
        pred = y_pred_val[i, :, 79 + z_max - 1 : 79 - 1 : -1].T
        # coarse = y_coarse_val[i, :, z_max - 1 :: -1].T
        # ref = y_ref_val[i, :, z_max - 1 :: -1].T
        # pred = y_pred_val[i, :, z_max - 1 :: -1].T
        print("coarse mse: ", np.std(coarse - ref))
        print("pred mse: ", np.std(pred - ref))

        # fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex=True, sharey=True)
        # ax[0].pcolormesh(coarse, vmin=ref.min(), vmax=ref.max())
        # ax[0].set_ylabel("coarse")
        # ax[1].pcolormesh(ref, vmin=ref.min(), vmax=ref.max())
        # ax[1].set_ylabel("reference")
        # ax[2].pcolormesh(pred, vmin=ref.min(), vmax=ref.max())
        # ax[2].set_ylabel("predicted")

        # fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex=True, sharey=True)
        # coarse_err = coarse - ref
        # pred_err = pred - ref
        # ax[0].pcolormesh(coarse_err, vmin=0, vmax=coarse_err.max())
        # ax[0].set_ylabel("coarse")
        # ax[2].pcolormesh(pred_err, vmin=0, vmax=coarse_err.max())
        # ax[2].set_ylabel("predicted")

        fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True, sharey=True)
        target_diff = np.diff(ref - coarse, axis=1)
        pred_diff = np.diff(pred - coarse, axis=1)
        im = ax[0].pcolormesh(target_diff)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_ylabel("target residual tendency")
        im = ax[1].pcolormesh(pred_diff, vmin=pred_diff.min(), vmax=pred_diff.max())
        plt.colorbar(im, ax=ax[1])
        ax[1].set_ylabel("predicted residual tendency")

        # coarse_diff = np.diff(coarse, axis=0)
        # ref_diff = np.diff(ref, axis=0)
        # pred_diff = np.diff(pred, axis=0)
        # fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex=True, sharey=True)
        # ax[0].pcolormesh(coarse_diff, vmin=ref_diff.min(), vmax=ref_diff.max())
        # ax[0].set_ylabel("coarse")
        # ax[1].pcolormesh(ref_diff, vmin=ref_diff.min(), vmax=ref_diff.max())
        # ax[1].set_ylabel("reference")
        # ax[2].pcolormesh(pred_diff, vmin=ref_diff.min(), vmax=ref_diff.max())
        # ax[2].set_ylabel("predicted")

        plt.show()
