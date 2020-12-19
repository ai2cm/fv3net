import argparse
import tempfile
import atexit
import os
import fsspec
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics

from fv3fit.keras import get_model_class
from loaders.batches import batches_from_serialized_callpyfort
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


def classification_metrics(y_true, y_pred):

    funcs = {
        "recall": skmetrics.recall_score,
        "precision": skmetrics.precision_score,
        "f1": skmetrics.f1_score,
    }
    out = {}
    for func_key, func in funcs.items():
        res = xr.apply_ufunc(
            func,
            y_true,
            y_pred,
            input_core_dims=[["sample"], ["sample"]],
            vectorize=True,
            kwargs=dict(zero_division=0),
            keep_attrs=True,
        )
        out[func_key] = res

    return out


def get_classification_scores(test_data, model, prob_thresh=0):

    test_data = _ThreadedSequencePreLoader(test_data)
    batch_metrics = []
    for batch in test_data:
        y_pred = model.predict(batch)
        y = batch[[var for var in y_pred]]
        # TODO: better way to tie this to user
        y_thresh = model.y_scaler.std.max() * 10 ** -4
        y_true = abs(y) > y_thresh
        y_pred = y_pred >= prob_thresh

        metrics = classification_metrics(y_true, y_pred)
        batch_metrics.append(metrics)

    result = {}
    for d in batch_metrics:
        for k, v in d.items():
            result.setdefault(k, []).append(v)

    for k, v in result.items():
        result[k] = xr.concat(v, dim="batch")

    return result


def get_target_predict_ds(sequential_batches, model, output_var):

    y_thresh = model.y_scaler.std.max() * 10**-4

    targets = xr.concat(
        [
            abs(batch[output_var]) >= y_thresh
            for batch in seq_batches
        ],
        dim="savepoint"
    )
    preds = xr.concat(
        [
            model.predict(batch)[output_var] >= 0
            for batch in seq_batches
        ],
        dim="savepoint"
    )

    return targets, preds


def group_fields_by_type(ds):

    scalars = ds[[var for var in ds if ds[var].ndim == 1]]
    vertical = ds[[var for var in ds if ds[var].ndim == 2]]

    return scalars, vertical


def plot_ens_spread_vert_field(da, ax=None, metric_name=None, title=None, xlim=None, label=None):

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
    (line,) = ax.plot(avg, levels, label=label)
    ax.fill_betweenx(levels, x1=lower, x2=upper, alpha=0.3, color=line.get_color())
    ax.set_ylabel("Vertical Level")
    ax.set_xlabel(metric_name)
    ax.set_xlim(xlim)
    ax.set_title(title)

    return plt.gcf()


def _fraction_true(da):
    return da.sum(dim="sample") / da.sizes["sample"]


def plot_true_val_ratios(targets, preds):

    fig, ax = plt.subplots()
    fig.set_dpi(120)
    fig.set_size_inches(3, 5)
    plot_ens_spread_vert_field(_fraction_true(targets), ax=ax, label="Target")
    plot_ens_spread_vert_field(_fraction_true(preds), ax=ax, label="Prediction")
    ax.legend()

    return fig


def plot_classify_over_time(da, seed=38):
    random = np.random.RandomState(seed)
    num_samples = da.sizes["sample"]
    idx = random.choice(range(num_samples), 20, replace=False)
    da.isel(sample=idx).plot.pcolormesh(
        x="time",
        y="lev",
        col="sample",
        col_wrap=2,
        add_colorbar=False,
    )
    fig = plt.gcf()
    return fig


def save_metrics(all_metrics, path):
    for metric_key, metrics in all_metrics.items():
        out_filename = f"{metric_key}.zarr"
        out_path = os.path.join(path, out_filename)
        metrics.to_zarr(out_path)


def _cleanup(tempdir: tempfile.TemporaryDirectory):
    tempdir.cleanup()


if __name__ == "__main__":
    args = parse_args()

    model = get_model_class("DenseClassifierModel")
    model = model.load(args.model_path)

    test_data = batches_from_serialized_callpyfort(args.test_data)
    # TODO add test range, currently 5 days
    test_data = test_data[(len(test_data) - 96 * 5):]
    # load 30-min sampled batches
    seq_batches = [batch for batch in test_data[:48:2]]
    shuffled = shuffle(test_data, seed=105)
    if args.num_test_batches is not None:
        shuffled = shuffled[:args.num_test_batches]

    # Metadata
    metadata = {
        "Model Source": args.model_path,
        "Test Data": args.test_data,
        "Test Batches Used": len(shuffled),
    }

    sample_batch = seq_batches[0]
    pred = model.predict(sample_batch)
    
    # Just a single output variable for a classifier for now
    output_var = list(pred.data_vars)[0]
    var_avg_spread = plot_ens_spread_vert_field(
        sample_batch.data_vars[output_var],
        title=output_var,
        metric_name=sample_batch[output_var].units,
    )

    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup, tmpdir)

    sections = {}
    insert_report_figure(
        sections,
        var_avg_spread,
        f"{output_var}_avg_spread.png",
        "Sample Variable Values",
        tmpdir.name,
    )

    metrics = get_classification_scores(shuffled, model)
    save_metrics(metrics, tmpdir.name)

    for mkey, metric_data in metrics.items():

        for var, da in metric_data.items():
            fig = plot_ens_spread_vert_field(da, metric_name=mkey, title=var)
            insert_report_figure(
                sections,
                fig,
                f"{var}_{mkey}.png",
                f"{var} vertical classification metrics",
                tmpdir.name,
            )

    targets, preds = get_target_predict_ds(seq_batches, model, output_var)
    # Can't serialize the multi-index
    # targets.reset_index().to_netcdf(os.path.join(tmpdir.name, "targets.nc"))
    # preds.reset_index().to_netcdf(os.path.join(tmpdir.name, "predictions.nc"))

    var_true_ratio = plot_true_val_ratios(targets, preds)
    insert_report_figure(
        sections,
        var_true_ratio,
        f"{output_var}_true_ratio.png",
        "True/False Ratio",
        tmpdir.name
    )

    classify_time_height_target = plot_classify_over_time(targets)
    insert_report_figure(
        sections,
        classify_time_height_target,
        f"{output_var}_target_time_height_targ.png",
        "Time x Height Classification Target",
        tmpdir.name
    )
    classify_time_height_pred = plot_classify_over_time(preds)
    insert_report_figure(
        sections,
        classify_time_height_pred,
        f"{output_var}_target_time_height_pred.png",
        "Time x Height Classification Prediction",
        tmpdir.name
    )

    report = create_html(sections, "Emulation Report", metadata=metadata,)

    with open(os.path.join(tmpdir.name, "emulation_classifier_report.html"), "w") as f:
        f.write(report)

    fs, _, _ = fsspec.get_fs_token_paths(args.output_folder)
    fs.put(tmpdir.name, args.output_folder, recursive=True)
