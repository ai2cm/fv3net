from train import load_window
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
    # n_blocks_train = 10 * blocks_per_day
    # n_blocks_val = 5 * blocks_per_day
    n_blocks_train = int(30 * blocks_per_day)
    n_blocks_val = int(7 * blocks_per_day)

    all_filenames = sorted(
        [os.path.join(block_dir, filename) for filename in os.listdir(block_dir)]
    )
    val_block_filenames = all_filenames[n_blocks_train : n_blocks_train + n_blocks_val]

    n_plots = 3
    idx = np.random.RandomState(3).choice(np.arange(48 * 48 * 6), size=[n_plots])
    X_coarse_val, y_coarse_val, y_coarse_delta_val, y_ref_val = load_window(
        idx, val_block_filenames
    )
    y_coarse_delta_val[:] *= 4
    # plt.figure()
    # im = plt.pcolormesh(y_coarse_delta_val[0, :, :].T)
    # plt.colorbar(im)
    # plt.figure()
    # # im = plt.pcolormesh((y_coarse_val[0, :, :] - y_ref_val[0, :, :]).T)
    # im = plt.pcolormesh(np.diff(y_coarse_val[0, :, :], axis=0).T)
    # plt.colorbar(im)
    # plt.figure()
    # im = plt.pcolormesh(np.diff(y_ref_val[0, :, :], axis=0).T)
    # plt.colorbar(im)
    # plt.show()

    y_coarse_delta_val[:, 0, :] += y_coarse_val[:, 0, :]

    vmin, vmax = 260, 300
    plt.figure()
    im = plt.pcolormesh(
        np.cumsum(y_coarse_delta_val[0, :, :], axis=0).T, vmin=vmin, vmax=vmax
    )
    plt.colorbar(im)
    plt.figure()
    # im = plt.pcolormesh((y_coarse_val[0, :, :] - y_ref_val[0, :, :]).T)
    im = plt.pcolormesh(y_coarse_val[0, :, :].T, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.figure()
    im = plt.pcolormesh(y_ref_val[0, :, :].T, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.show()
    # y_coarse_delta_val[:] = 0.
    y_coarse_no_nudge_val = np.cumsum(y_coarse_delta_val, axis=1)
    for i in range(n_plots):

        z_max = 79
        coarse = y_coarse_val[i, :, 79 + z_max - 1 : 79 - 1 : -1].T
        coarse_no_nudge = y_coarse_no_nudge_val[i, :, 79 + z_max - 1 : 79 - 1 : -1].T
        coarse_no_nudge_truediff = y_coarse_delta_val[
            i, 0:, 79 + z_max - 1 : 79 - 1 : -1
        ].T
        ref = y_ref_val[i, :, 79 + z_max - 1 : 79 - 1 : -1].T

        # coarse = np.diff(coarse, axis=1)
        # coarse_no_nudge = np.diff(coarse_no_nudge, axis=1)
        # ref = np.diff(ref, axis=1)

        # coarse = y_coarse_val[i, :, z_max - 1 :: -1].T
        # ref = y_ref_val[i, :, z_max - 1 :: -1].T

        print("coarse mse: ", np.std(coarse - ref))

        fig, ax = plt.subplots(3, 1, figsize=(8, 5), sharex=True, sharey=True)
        ax[0].pcolormesh(coarse, vmin=ref.min(), vmax=ref.max())
        ax[0].set_ylabel("coarse")
        ax[1].pcolormesh(coarse_no_nudge, vmin=ref.min(), vmax=ref.max())
        ax[1].set_ylabel("no nudge")
        ax[2].pcolormesh(ref, vmin=ref.min(), vmax=ref.max())
        ax[2].set_ylabel("reference")
        plt.colorbar(im)

        # fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex=True, sharey=True)
        # coarse_err = coarse - ref
        # ax[0].pcolormesh(coarse_err, vmin=0, vmax=coarse_err.max())
        # ax[0].set_ylabel("coarse")

        # fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True, sharey=True)
        # target_diff = np.diff(ref - coarse, axis=1)
        # im = ax[0].pcolormesh(target_diff)
        # plt.colorbar(im, ax=ax[0])
        # ax[0].set_ylabel("target residual tendency")

        coarse_diff = np.diff(coarse, axis=0)
        coarse_no_nudge_diff = np.diff(coarse_no_nudge, axis=0)
        ref_diff = np.diff(ref, axis=0)
        vmin = coarse_no_nudge_diff.min()
        vmax = coarse_no_nudge_diff.max()
        fig, ax = plt.subplots(3, 1, figsize=(8, 5), sharex=True, sharey=True)
        ax[0].pcolormesh(coarse_diff, vmin=vmin, vmax=vmax)
        ax[0].set_ylabel("coarse")
        ax[1].pcolormesh(
            np.cumsum(coarse_no_nudge_truediff, axis=1), vmin=vmin, vmax=vmax
        )
        ax[1].set_ylabel("coarse no nudge")
        ax[2].pcolormesh(coarse_no_nudge_truediff, vmin=vmin, vmax=vmax)
        ax[2].set_ylabel("coarse no nudge true")
        # ax[2].pcolormesh(ref_diff, vmin=vmin, vmax=vmax)
        # ax[2].set_ylabel("reference")

        plt.show()
