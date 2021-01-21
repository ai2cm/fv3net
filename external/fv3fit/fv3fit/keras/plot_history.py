import argparse
import fsspec
import json
import logging
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import tempfile
from typing import Sequence

from . import History
from vcm.cloud import gsutil


logger = logging.getLogger(__name__)


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _plot_loss(
    loss_history: Sequence[float], val_loss_history=None, xlabel="epoch"
) -> plt.Figure:
    x = range(len(loss_history))
    fig = plt.figure()
    plt.plot(x, loss_history, "-", label="loss")
    if val_loss_history:
        plt.plot(x, val_loss_history, "--", label="validation loss")

    plt.xlabel(xlabel)
    plt.ylabel("loss")
    plt.legend()
    return fig


def _plot_loss_per_batch(history: History) -> plt.Figure:
    n_epochs = len(history["loss"])
    loss = history["loss"]
    val_loss = history.get("val_loss", [])

    fig = plt.figure(figsize=(8, 3 * n_epochs))
    fig.subplots_adjust(hspace=0)
    y_range = (
        0.95 * np.min(_flatten(loss) + _flatten(val_loss)),
        1.05 * np.max(_flatten(loss) + _flatten(val_loss)),
    )
    for i_epoch in range(n_epochs):
        x = range(len(history["loss"][i_epoch]))
        ax = fig.add_subplot(n_epochs, 1, i_epoch + 1)
        ax.plot(x, history["loss"][i_epoch], "-", label="loss")
        if "val_loss" in history:
            ax.plot(x, history["val_loss"][i_epoch], "--", label="validation loss")
        ax.set_ylim(y_range)
        ax.text(
            0.9,
            0.1,
            f"epoch {i_epoch}",
            horizontalalignment="right",
            transform=ax.transAxes,
        )
        ax.set_ylabel("loss")

    ax.set_xlabel("batch")
    ax.legend()
    return fig


def _copy_outputs(temp_dir, output_dir) -> None:
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(temp_dir, output_dir)


def _get_epoch_losses(history: History, key: str):
    if key not in history:
        return None
    if key == "val_loss":
        return [epoch_batch_losses[-1] for epoch_batch_losses in history[key]]
    else:
        return [np.mean(epoch_batch_losses) for epoch_batch_losses in history[key]]


def _plot_training_history(history_file: str, output_dir: str) -> None:
    with fsspec.open(history_file, "r") as f:
        history = json.load(f)
    loss_at_epoch_end = _get_epoch_losses(history, "loss")
    val_loss_at_epoch_end = _get_epoch_losses(history, "val_loss")
    loss_saved_per_batch = True if len(history["loss"][0]) > 1 else False
    with tempfile.TemporaryDirectory() as tmpdir:
        _plot_loss(loss_at_epoch_end, val_loss_at_epoch_end).savefig(
            os.path.join(tmpdir, "loss_over_epochs.png")
        )
        if loss_saved_per_batch:
            _plot_loss_per_batch(history).savefig(
                os.path.join(tmpdir, "epoch_losses_over_batches.png")
            )
        _copy_outputs(tmpdir, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "history-path", type=str, help="Path of training history json file."
    )
    parser.add_argument(
        "output-dir", type=str, help="Path for saving figures",
    )


if __name__ == "__main__":
    args = parse_args()
    _plot_training_history(args.history_path, args.output_dir)
    logger.info(f"Saved keras training history figures to {args.output_dir}")
