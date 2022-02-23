# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: fv3net
#     language: python
#     name: fv3net
# ---

import os

import fsspec
import fv3fit
import tensorflow as tf
import vcm.catalog
import vcm.fv3
import xarray
from fv3fit.emulation.compositions import apply_difference
from fv3fit.keras.adapters import merge_models

import fv3net.artifacts.resolve_url
import wandb


def get_in(inputs):
    qv_in = inputs["specific_humidity_input"]
    t_in = inputs["air_temperature_input"]
    qc_in = inputs["cloud_water_mixing_ratio_input"]
    return qv_in, t_in, qc_in


def get_gscond(inputs):
    qv_g = inputs["specific_humidity_after_gscond"]
    t_g = inputs["air_temperature_after_gscond"]
    return qv_g, t_g


def get_precpd(inputs):
    qv_in = inputs["specific_humidity_after_precpd"]
    t_in = inputs["air_temperature_after_precpd"]
    qc_in = inputs["cloud_water_mixing_ratio_after_precpd"]
    return qv_in, t_in, qc_in


def get_outputs(qv, t, qvp, tp, qcp):
    return {
        "specific_humidity_after_gscond": qv,
        "air_temperature_after_gscond": t,
        "specific_humidity_after_precpd": qvp,
        "air_temperature_after_precpd": tp,
        "cloud_water_mixing_ratio_after_precpd": qcp,
    }


def enforce_positive(d):
    return {name: tf.keras.layers.ReLU(name=name)(x) for name, x in d.items()}


def mask_outputs(mask, inputs, outputs):
    qv, t, qc = get_in(inputs)
    qvg, tg = get_gscond(outputs)
    qvp, tp, qcp = get_precpd(outputs)

    # mask where initial physics is always 0
    qvg_m = tf.where(mask, qv, qvg)
    tg_m = tf.where(mask, t, tg)
    qvp_m = tf.where(mask, qv, qvp)
    tp_m = tf.where(mask, t, tp)
    qcp_m = tf.where(mask, qc, qcp)

    # ensure positive
    return get_outputs(qvg_m, tg_m, qvp_m, tp_m, qcp_m)


def fix_model(model, mask):
    inputs = {in_.name: in_ for in_ in model.inputs}
    outputs = model(inputs)
    # outputs.update(mask_outputs(mask, inputs, outputs))
    positive_outputs = enforce_positive(outputs)
    return tf.keras.Model(inputs=inputs, outputs=positive_outputs)


def get_mask() -> tf.Tensor:
    url = "gs://vcm-ml-experiments/microphysics-emulation/2022-02-08/rnn-alltdep-47ad5b-login-6h-v2-online"  # noqa
    ds = xarray.open_zarr(
        fsspec.get_mapper(url + "/atmos_dt_atmos.zarr"), consolidated=True
    )
    piggy = xarray.open_zarr(fsspec.get_mapper(url + "/piggy.zarr"), consolidated=True)
    grid = vcm.catalog.catalog["grid/c48"].to_dask()

    ds = vcm.fv3.gfdl_to_standard(xarray.merge([ds, piggy], compat="override")).assign(
        **grid
    )

    # There are unrealistic non-zero predictions in the stratosphere.

    # # Save a model which enforces positivity and zeros prediction in upper atmosphere

    fraction_pos = (
        (ds.tendency_of_cloud_water_due_to_zhao_carr_physics > 0)
        .isel(time=0)
        .groupby("z")
        .mean(xarray.ALL_DIMS)
    )
    fraction_pos.drop("z").plot()

    # reverse mask since k = 0 is bottom when model is run online
    mask = tf.constant((fraction_pos.values == 0)[::-1])
    return mask


def save_model(run, model):
    new_model_root = fv3net.artifacts.resolve_url.resolve_url(
        "vcm-ml-experiments", "microphysics-emulation", run.id
    )
    print(new_model_root)

    with fv3fit._shared.filesystem.put_dir(new_model_root) as d:
        model.save(os.path.join(d, "model.tf"))

    art = wandb.Artifact("post-processed-model", type="model")
    art.add_reference(new_model_root)
    wandb.log_artifact(art)
    wandb.finish()


def compose(precpd: str, gscond: str):
    run = wandb.init(project="microphysics-emulation", entity="ai2cm")
    # open models
    artifact = run.use_artifact(precpd, type="model",)
    precpd_model = tf.keras.models.load_model(artifact.download())

    artifact = run.use_artifact(gscond, type="model")
    gscond_model = tf.keras.models.load_model(artifact.download())
    # -

    precpd_model.summary()

    gscond_model.summary()

    # +

    gscond_model._name = "gscond"
    precpd_model._name = "precpd"
    combined_model = merge_models(precpd_model, gscond_model)
    final_model = apply_difference(combined_model)
    # -

    final_model.summary()

    # +
    mask = get_mask()
    new_model = fix_model(final_model, mask)
    # -

    new_model.summary()

    new_model(new_model.input)
    save_model(run, new_model)


def mask(model_path: str):
    run = wandb.init(project="microphysics-emulation", entity="ai2cm")
    artifact = run.use_artifact(model_path, type="model")
    model = tf.keras.models.load_model(artifact.download())

    model.summary()
    difference_model = apply_difference(model)
    mask = get_mask()
    new_model = fix_model(difference_model, mask)
    new_model.summary()
    new_model(new_model.input)
    save_model(run, new_model)


mask("ai2cm/microphysics-emulation/microphysics-emulator-dense-local:v1")
