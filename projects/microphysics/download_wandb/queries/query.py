# flake8: noqa
from collections import defaultdict
import pandas as pd
import networkx as nx
import datetime
import sqlite3
import json
import dataclasses
import dacite
from typing import Optional, List
from enum import Enum
from pymonad.maybe import Maybe, Just, Nothing


class JobType(Enum):
    train = "train"
    prognostic_run = "prognostic_run"
    piggy = "piggy-back"
    prognostic_evaluation = "prognostic_evaluation"
    _test = "test"
    figures = "figures"
    netcdf_gather = "netcdf-gather"
    # defunct names
    _training_run = "training-run"
    _training = "training"
    _prognostic = "prognostic"
    train_score = "train_score"


@dataclasses.dataclass
class Run:
    id: str
    state: str
    name: str
    config: dict
    summary: dict
    job_type: Optional[JobType]
    tags: List[str]
    created_at: datetime.datetime
    group_: Optional[str]

    @classmethod
    def from_dict(cls, d):
        return dacite.from_dict(
            Run,
            d,
            config=dacite.Config(
                type_hooks={
                    datetime.datetime: datetime.datetime.fromisoformat,
                    dict: json.loads,
                    List[str]: json.loads,
                },
                cast=[JobType],
            ),
        )


path = "projects/microphysics/download_wandb/example.db"
db = sqlite3.connect(path)
db.row_factory = sqlite3.Row

cur = db.cursor()

# ['id', 'state', 'name', 'config', 'summary', 'job_type', 'tags', 'created_at', 'group_']
runs = list(cur.execute("SELECT * FROM runs;"))
db.close()


# %%
print(runs[0])

G = nx.DiGraph()

for obj in runs:
    run = Run.from_dict(obj)
    G.add_node(run.id, run=run)


def lookup(d, keys) -> Maybe:
    val = d
    try:
        for key in keys:
            val = val[key]
        return Just(val)
    except (KeyError, TypeError):
        return Nothing


def exactly_on(maybes: List[Maybe]) -> Maybe:
    justs = [maybe.maybe(0, lambda x: x) for maybe in maybes if maybe.is_just()]

    if len(justs) != 1:
        return Nothing
    else:
        return Just(justs[0])


def _get_model_1(d):
    return exactly_on(
        [
            lookup(d, ["model", "value"]),
            lookup(d, ["config", "value", "zhao_carr_emulation", "model", "path"]),
            lookup(d, ["config", "value", "zhao_carr_emulation", "gscond", "path"]),
        ]
    )


def _get_model_url(d):
    return lookup(d, ["out_url", "value"])


assert _get_model_1({"model": {"value": "adfa"}}) == Just("adfa")
val = _get_model_1(
    {"config": {"value": {"zhao_carr_emulation": {"model": {"path": "adfa"}}}}}
)
assert val == Just("adfa"), val


def prog_connected(train: Run, prog: Run):
    return (
        _get_model_url(train.config)
        .bind(
            lambda model_url: _get_model_1(prog.config).bind(
                lambda prog_url: Just(prog_url.startswith(model_url))
            )
        )
        .maybe(False, lambda x: x)
    )


def piggy_connected(prog: Run, piggy: Run) -> bool:
    return (
        lookup(prog.config, ["rundir", "value"]).bind(
            lambda prog_rundir: lookup(piggy.config, ["run", "value"]).bind(
                lambda piggy_run: Just(prog_rundir == piggy_run)
            )
        )
    ).maybe(False, lambda x: x)


def test_connected(train: Run, test: Run) -> bool:
    try:
        return train.config["out_url"]["value"] == test.config["out_url"]["value"]
    except KeyError:
        return False


connections = defaultdict(lambda: (lambda x, y: False))
connections[(JobType.train, JobType.prognostic_run)] = prog_connected
connections[(JobType.prognostic_run, JobType.piggy)] = piggy_connected
connections[(JobType.train, JobType.train_score)] = test_connected

nodes = G.nodes

for node in nodes:
    node = G.nodes[node]
    in_run: Run = node["run"]
    for other_node in nodes:
        other_node = nodes[other_node]
        other_run: Run = other_node["run"]
        if other_run.created_at >= in_run.created_at:
            connected = connections[(in_run.job_type, other_run.job_type)](
                in_run, other_run
            )
            if connected:
                G.add_edge(in_run.id, other_run.id)


def find_path(G, nodes, first_type, *types):
    for node in nodes:
        run = G.nodes[node]["run"]
        if run.job_type == first_type:
            ret_value = G.nodes[node]["run"]
            if len(types) == 0:
                yield (ret_value,)
            else:
                kids = G.successors(node)
                for path in find_path(G, kids, *types):
                    yield (ret_value, *path)


def justs(a):
    return [x.maybe(0, lambda x: x) for x in a if x.is_just()]


def lookup_piggy(G):
    def _gen():
        for (train, prog, piggy) in find_path(
            G, G.nodes, JobType.train, JobType.prognostic_run, JobType.piggy
        ):

            is_bug = "bug" in train.tags + prog.tags + piggy.tags

            gscond_only = prog.config["config"]["value"]["namelist"][
                "gfs_physics_nml"
            ].get("emulate_zc_only", False)
            online = prog.config["config"]["value"]["namelist"]["gfs_physics_nml"].get(
                "emulate_zc_microphysics", False
            )

            mode = {
                (True, False): "offline",
                (False, False): "offline",
                (True, True): "gscond_only",
                (False, True): "online",
            }[(gscond_only, online)]
            try:
                duration_seconds = piggy.summary["duration_seconds"]
            except KeyError:
                duration_seconds = None

            yield {
                "id": train.id,
                "created_at": train.created_at,
                "model_name": train.name,
                "duration_seconds": duration_seconds,
                "is_bug": is_bug,
                "mode": mode,
                "group": prog.group_,
                **train.summary,
            }

    return pd.DataFrame.from_records(_gen())


df = lookup_piggy(G)
df["duration_days"] = df.duration_seconds / 86400
out = df[df["mode"] == "online"][
    [
        "duration_days",
        "is_bug",
        "group",
        "model_name",
        "val_cloud_water_mixing_ratio_after_precpd_loss",
    ]
]
print(out)
out.to_csv("leaders.csv")
