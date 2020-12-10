import json
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import tempfile
from typing import Mapping

from vcm.cloud import gsutil

TRAINING_LOG_FILENAME = "training_history.json"


def _plot_loss(loss_history, val_loss_history=None, xlabel="epoch") -> plt.Figure:
    x = range(len(loss_history))
    fig = plt.figure()
    plt.plot(x, loss_history, "-", label="loss")
    if val_loss_history:
        plt.plot(x, val_loss_history, "--", label="validation loss")
    
    plt.xlabel(xlabel)
    plt.ylabel("loss")
    plt.legend()
    return fig


def _plot_loss_per_batch(history) -> plt.Figure:
    if isinstance(history["loss"][0], list):
        n_epochs = len(history["loss"]) 
    else:
        raise ValueError("Can only plot loss over batches if num_batches was supplied as a fit kwarg.")
    fig = plt.figure(figsize=(8, 3 * n_epochs))
    fig.subplots_adjust(hspace=0)
    y_range = (
        0.95 * np.min(history["loss"] + history["val_loss"]),
        1.05 * np.max(history["loss"] + history["val_loss"]))
    for i_epoch in range(n_epochs):
        x = range(len(history["loss"][i_epoch]))
        ax = fig.add_subplot(n_epochs, 1, i_epoch + 1)
        ax.plot(x, history["loss"][i_epoch], "-", label="loss")
        if "val_loss" in history:
            ax.plot(x, history["val_loss"][i_epoch], "--", label="validation loss")
        ax.set_ylim(y_range)
        ax.text(0.9, 0.1, f"epoch {i_epoch}", horizontalalignment='right', transform=ax.transAxes)
        ax.set_ylabel("loss")
        
    ax.set_xlabel("batch")
    ax.legend()
    return fig


def _copy_outputs(temp_dir, output_dir) -> None:
    if output_dir.startswith("gs://"):
        gsutil.copy_directory_contents(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir, dirs_exist_ok=True)


def save_history(history: Mapping, output_dir: str) -> None:
    # if fit with batch_size fit kwarg, will save the loss within each epoch
    # as .fit is called on each batch in the sequence
    loss_saved_per_batch = True if isinstance(history["loss"][0], list) else False
    if loss_saved_per_batch:
        loss_at_epoch_end = [epoch_batch_losses[-1] for epoch_batch_losses in history["loss"]]
        val_loss_at_epoch_end = [epoch_batch_losses[-1] for epoch_batch_losses in history["val_loss"]]
    else:
        loss_at_epoch_end = history["loss"]
        val_loss_at_epoch_end = history["val_loss"]
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, TRAINING_LOG_FILENAME), "w") as f:
            json.dump(history, f)
        _plot_loss(loss_at_epoch_end, val_loss_at_epoch_end).savefig(os.path.join(tmpdir, "loss_over_epochs.png"))
        if loss_saved_per_batch:
            _plot_loss_per_batch(history).savefig(os.path.join(tmpdir, "epoch_losses_over_batches.png"))
        _copy_outputs(tmpdir, output_dir)
        