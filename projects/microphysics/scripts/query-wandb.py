#!/usr/bin/env python3
import json
import typer
import wandb
import db

app = typer.Typer()

PROJECT = "ai2cm/microphysics-emulation"


@app.command()
def groups(experiment: str, filter_tags: str = typer.Option("", "-f")):
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"tags": experiment})
    filter_tags = set(filter_tags.split(","))
    groups = set(run.group for run in runs if len(filter_tags & set(run.tags)) == 0)
    for group in groups:
        print(group)


@app.command()
def prognostic_runs(experiment: str, filter_tags: str = typer.Option("", "-f")):
    """Show the top level metrics for prognostic runs tagged by `experiment`

    Examples:

    Rerun the piggy-backed diagnostics for all runs in an experiment::

        top.py prognostic-runs experiment/squash -f bug \
            | parallel conda run -n fv3net prognostic_run_diags piggy -s

    """
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"tags": experiment})
    db.insert_runs(runs)
    filter_tags = tuple(set(filter_tags.split(",")))
    groups_query = db.query(
        """
    SELECT group_
    FROM (
        SELECT *, max(json_each.value in (?)) as bug
        FROM runs, json_each(tags)
        WHERE job_type='prognostic_run'
        GROUP BY runs.id
    )
    WHERE not bug
    """,
        filter_tags,
    )

    # show metrics
    for group in groups_query:
        stats = query_top_level_metrics(group)
        print(json.dumps(stats))


def query_top_level_metrics(group):
    cur = db.query(
        """
        SELECT
            group_ as "group",
            max(json_extract(summary, "$.duration_seconds")) as duration_seconds,
            max(json_extract(summary, "$.global_average_cloud_5d_300mb_ppm"))
                as global_average_cloud_5d_300mb_ppm
        FROM runs
        WHERE group_ = ?
    """,
        group,
    )
    keys = [it[0] for it in cur.description]
    (row,) = cur
    return dict(zip(keys, row))


@app.command()
def runs(experiment: str, filter_tags: str = typer.Option("", "-f")):
    """
    Examples:

    Filter all runs with "bug" in tags::

        ./top.py runs experiment/squash -f bug | jq -sr '[.[].group] | unique | .[]'
    """
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"tags": experiment})
    filter_tags = set(filter_tags.split(","))
    for run in runs:
        if len(filter_tags & set(run.tags)) == 0:
            summary = {}
            for k, v in run.summary.items():
                # ensure that summary can be serialized to json
                try:
                    json.dumps(v)
                except Exception:
                    pass
                else:
                    summary[k] = v

            d = {
                "job_type": run.job_type,
                "group": run.group,
                "tags": run.tags,
                "id": run.id,
                "url": run.url,
                "summary": summary,
                "config": run.config,
            }
            print(json.dumps(d))


@app.command()
def tags():
    api = wandb.Api()
    runs = api.runs(PROJECT)
    tags = set.union(*(set(run.tags) for run in runs))
    for tag in tags:
        print(tag)


if __name__ == "__main__":
    app()
