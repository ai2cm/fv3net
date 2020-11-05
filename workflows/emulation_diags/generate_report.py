import argparse
import tempfile
import atexit
import os
import fsspec
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

from fv3fit.keras import get_model_class
from loaders.batches import batches_from_serialized
from loaders import shuffle
from report import insert_report_figure, create_html

# TODO move into public usage api
from fv3fit.keras._models._sequences import _ThreadedSequencePreLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", type=str, help="Path to emulation model")
    parser.add_argument("test_data", type=str, help="Path to test data")
    parser.add_argument("output_folder", type=str, help="Path to save model figures")
    parser.add_argument("--num-test-batches", required=False, default=None, type=int)

    return parser.parse_args()


def plot_targ_pred_scatter(target, prediction, ax, title=""):
    ymax = target.max().values
    ymin = target.min().values
    plt.hexbin(
        target.values,
        prediction.values,
        gridsize=50,
        cmap="Purples",
        norm=mpl_colors.LogNorm(),
        extent=[ymin, ymax, ymin, ymax],
    )
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
    mse = (diff ** 2).mean(dim="sample")
    rmse = xr.ufuncs.sqrt(mse)
    ss_tot = ((target - target.mean(dim="sample")) ** 2).sum(dim="sample")
    ss_res = ((diff) ** 2).sum(dim="sample")
    r2 = 1 - ss_res / ss_tot
    return {"rmse": rmse, "r2": r2, "bias": bias}


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

    rmse = xr.concat(by_metric["rmse"], dim="batch")
    r2 = xr.concat(by_metric["r2"], dim="batch")
    bias = xr.concat(by_metric["bias"], dim="batch")

    return {"rmse": rmse, "r2": r2, "bias": bias}


def group_fields_by_type(ds):

    scalars = ds[[var for var in ds if ds[var].ndim == 1]]
    vertical = ds[[var for var in ds if ds[var].ndim == 2]]

    return scalars, vertical


def plot_ens_spread_vert_field(da, ax=None, metric_name=None, title=None, xlim=None):

    if da.ndim != 2:
        raise ValueError(
            "Plot ensemble spread expects  2D data (sample x vertical_level)"
        )

    field_arr = da.values
    upper = np.percentile(field_arr, 97.5, axis=0)
    lower = np.percentile(field_arr, 2.5, axis=0)
    avg = field_arr.mean(axis=0)

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(2.5, 5)
        fig.set_dpi(120)

    levels = np.arange(0, field_arr.shape[1])
    (line,) = ax.plot(avg, levels)
    ax.fill_betweenx(levels, x1=lower, x2=upper, alpha=0.3, color=line.get_color())
    ax.set_ylabel("Vertical Level")
    ax.set_xlabel(metric_name)
    ax.set_xlim(xlim)
    ax.set_title(title)

    return fig


def save_metrics(all_metrics, path):
    for metric_key, metrics in all_metrics.items():
        out_filename = f"{metric_key}.nc"
        out_path = os.path.join(path, out_filename)
        metrics.to_netcdf(out_path)


def parse_metrics_for_table(all_metrics):

    metric_columns = {
        "Mean": ("mean", ()),
        "Median": ("median", ()),
        "Percentile (2.5%)": ("quantile", (0.025,)),
        "Percentile (97.5%)": ("quantile", (0.975,)),
        "Min": ("min", ()),
        "Max": ("max", ()),
    }

    metrics_for_table = {}
    for unique_idx, (metric_key, metric_values) in enumerate(all_metrics.items()):

        # Add an empty row header for metric type separation
        metrics_for_table[metric_key] = {colname: "" for colname in metric_columns}

        # Calc stats for each metric
        for colname, (func_name, func_args) in metric_columns.items():
            func = getattr(metric_values, func_name)
            result = func(*func_args)

            # Add metric stats for all variables
            for var_name, metric_result in result.items():
                unique_name = var_name + f"_{unique_idx}"
                result_str = f"{metric_result.values.item():1.2e}"
                metrics_for_table.setdefault(unique_name, {})[colname] = result_str

    return metrics_for_table


def _cleanup(tempdir: tempfile.TemporaryDirectory):
    tempdir.cleanup()


if __name__ == "__main__":
    args = parse_args()

    model = get_model_class("DenseModel")
    model = model.load(args.model_path)

    test_data = batches_from_serialized(args.test_data)
    # TODO add test range, currently 5 days
    test_data = shuffle(test_data[(len(test_data) - 96 * 5):], seed=105)
    if args.num_test_batches is not None:
        test_range = len(test_data) - args.num_test_batches
        test_data = test_data[test_range:]

    # Metadata
    metadata = {
        "Model Source": args.model_path,
        "Test Data": args.test_data,
        "Test Batches Used": len(test_data),
    }

    batches_for_targ_v_pred = 5
    loaded = [test_data[i] for i in range(batches_for_targ_v_pred)]
    predictions = [model.predict(test_batch) for test_batch in loaded]
    predictions = xr.concat(predictions, "sample")
    targets = xr.concat(loaded, "sample")

    targ_pred_figs = get_targ_pred_figs(targets, predictions)

    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup, tmpdir)

    sections = {}
    for fig, var in zip(targ_pred_figs, predictions):
        insert_report_figure(
            sections,
            fig,
            f"targ_v_pred_{var}.png",
            f"Sample Target vs Prediction (num_batches: {batches_for_targ_v_pred})",
            tmpdir.name,
        )
        plt.close(fig)

    metrics = get_emulation_metrics(test_data, model)
    save_metrics(metrics, tmpdir.name)

    for mkey, metric_data in metrics.items():
        scalars, verticals = group_fields_by_type(metric_data)

        if "r2" in mkey:
            xlim = (-2, 1)
        else:
            xlim = None

        for var, da in verticals.items():
            fig = plot_ens_spread_vert_field(da, metric_name=mkey, title=var, xlim=xlim)
            insert_report_figure(
                sections,
                fig,
                f"{var}_{mkey}.png",
                f"{var} vertical metrics",
                tmpdir.name,
            )

    metric_stats = parse_metrics_for_table(metrics)

    report = create_html(
        sections,
        "Emulation Report",
        metrics=metric_stats,
        metadata=metadata,
        html_header="lol",
    )
    with open(os.path.join(tmpdir.name, "emulation_report.html"), "w") as f:
        f.write(report)

    fs, _, _ = fsspec.get_fs_token_paths(args.output_folder)
    fs.put(tmpdir.name, args.output_folder, recursive=True)
