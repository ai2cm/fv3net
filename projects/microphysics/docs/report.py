# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: fv3net
#     language: python
#     name: fv3net
# ---

# %% [markdown]
# Markdown with math $x=y$.

# %%
import fv3config
import pandas as pd
import json
import datetime

import wandb
from collections import defaultdict
import toolz
import plotly.express as px


def is_online(group):
    return group["prognostic_run"].config["config"]["namelist"]["gfs_physics_nml"][
        "emulate_zc_microphysics"
    ]


def run_duration(group):
    return fv3config.get_run_duration(group["prognostic_run"].config["config"])


@toolz.memoize(key=lambda args, kwargs: (args[0].id, args[1]))
def open_table(run, key):
    f = run.file(run.summary[key]["path"])
    fobj = f.download(replace=True)
    data = json.load(fobj)
    return pd.DataFrame(data["data"], columns=data["columns"])


def get_model_url(group):
    return group["prognostic_run"].config["env"]["TF_MODEL_PATH"]


def get_surface_precip_skill(group):
    return open_table(group["piggy-back"], "skill_time")


def get_data(group, online=True):

    groups = [
        val
        for group, val in group.items()
        if "7166bd" in group
        and online == is_online(val)
        and "piggy-back" in val
        and run_duration(val) == datetime.timedelta(days=10)
    ]
    return pd.concat(
        [
            get_surface_precip_skill(group).assign(
                model_url=get_model_url(group),
                model_tag=get_model_url(group).split("/")[-2],
                online=online,
            )
            for group in groups
        ]
    )


api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("ai2cm/microphysics-emulation", filters={"state": "finished"})
group = defaultdict(dict)
summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files

    if run.group is not None:
        group[run.group][run.job_type] = run
group = dict(group)

# %%
merged = pd.concat([get_data(group, online=b) for b in [True, False]])

# %%

wong_palette = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

# plotly express
px.defaults.color_discrete_sequence = wong_palette

# %%
px.line(
    merged.sort_values(["model_tag", "time"]),
    x="time",
    y="surface_precipitation",
    color="model_tag",
    facet_col="online",
)

# %%
px.bar(
    merged.groupby(["model_tag", "online"])
    .mean()
    .reset_index()
    .sort_values("model_tag"),
    x="online",
    y="surface_precipitation",
    color="model_tag",
    barmode="group",
    width=600,
)
