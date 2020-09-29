import argparse
import tempfile
import atexit
import os
import fsspec
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

from fv3fit.keras import get_model_class
from loaders.batches import batches_from_serialized
from report import insert_report_figure, create_html

# TODO move into public usage api
from fv3fit.keras._models._sequences import _ThreadedSequencePreLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to emulation model"
    )
    parser.add_argument(
        "test_data",
        type=str,
        help="Path to test data"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to save model figures"
    )
    parser.add_argument(
        "--num-test-batches",
        required=False,
        default=None,
        type=int
    )

    return parser.parse_args()


def plot_targ_pred_scatter(target, prediction, ax, title=""):
    ymax = target.max().values
    ymin = target.min().values
    plt.hexbin(
        target.values,
        prediction.values,gridsize=50, cmap="Purples", norm=mpl_colors.LogNorm(), extent=[ymin, ymax, ymin, ymax])
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.set_title(f"{target.long_name} {title}")
    plt.colorbar()


def get_targ_pred_figs(target_ds, pred_ds):

    figs = []
    for var, prediction in pred_ds.items():
        target = target_ds[var]
        fig, ax = plt.subplots()
        plot_targ_pred_scatter(target, prediction, ax)
        figs.append(fig)

    return figs


def _calc_metrics(target, prediction):
    diff = prediction - target
    bias = diff.mean(dim="sample")
    mse = (diff**2).mean(dim="sample")
    ss_tot = ((target - target.mean(dim="sample"))**2).sum(dim="sample")
    ss_res = ((diff)**2).sum(dim="sample")
    r2 = 1 - ss_res / ss_tot
    return {"mse": mse, "r2": r2, "bias": bias}


def get_emulation_metrics(test_data, model):

    test_data = _ThreadedSequencePreLoader(test_data)
    metrics = []
    for target in test_data:
        prediction = model.predict(target)
        metrics.append(_calc_metrics(target, prediction))
    
    by_metric = {}
    for d in metrics:
        for k, v in d.items():
            by_metric.setdefault(k, []).append(v)

    mse = xr.concat(by_metric["mse"], dim="batch")
    r2 = xr.concat(by_metric["r2"], dim="batch")
    bias = xr.concat(by_metric["bias"], dim="batch")

    return {"mse": mse, "r2": r2, "bias": bias}


def _cleanup(tempdir: tempfile.TemporaryDirectory):
    tempdir.cleanup()


if __name__ == "__main__":
    args = parse_args()

    model = get_model_class("DenseModel")
    model = model.load(args.model_path)

    test_data = batches_from_serialized(args.test_data)
    if args.num_test_batches is not None:
        test_range = len(test_data) - args.num_test_batches
        test_data = test_data[test_range:]

    predictions = [model.predict(test_data[i]) for i in range(5)]
    predictions = xr.concat(predictions, "sample")
    targets = xr.concat([test_data[i] for i in range(5)], "sample")

    targ_pred_figs = get_targ_pred_figs(targets, predictions)

    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup, tmpdir)

    sections = {}
    for fig, var in zip(targ_pred_figs, predictions):
        insert_report_figure(
            sections,
            fig,
            f"targ_v_pred_{var}.png",
            "Sample Target vs Prediction",
            tmpdir.name
        )
        plt.close(fig)

    metrics = get_emulation_metrics(test_data, model)

    report = create_html(sections, "Emulation Results")
    with open(os.path.join(tmpdir.name, "emulation_report.html"), "w") as f:
        f.write(report)

    fs, _, _ = fsspec.get_fs_token_paths(args.output_folder)
    fs.put(tmpdir.name, args.output_folder)
