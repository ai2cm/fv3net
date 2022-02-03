import sqlite3
import wandb
import os
import logging
import tqdm
import time_to_crash

project = "microphysics-emulation"
entity = "ai2cm"

api = wandb.Api()


def run_from_id(wandb_id):
    return api.run(os.path.join(entity, project, wandb_id))


def get_rundir_from_prog_run(run):
    artifacts = list(run.logged_artifacts())
    art = artifacts[0]
    config_path = "fv3config.yml"
    config_file = art.get_path(config_path)
    rundir = config_file.ref_target()[: -len(config_path)]
    return rundir


con = sqlite3.connect("example.db")

cur = con.cursor()
count = list(cur.execute("select count(*) from prognostic_runs"))[0][0]
print(count)

cur = con.cursor()
out = cur.execute("select id, wandb_id from prognostic_runs")
for row in tqdm.tqdm(out, total=count):
    id_, wandb_id = row
    run = run_from_id(wandb_id)
    try:
        rundir = get_rundir_from_prog_run(run)
        duration = time_to_crash.get_duration(rundir)
        cur1 = con.cursor()
        cur1.execute(
            """
            INSERT INTO progsummary (run_id, location, duration_seconds)
            VALUES (?, ?, ?)
            """,
            (id_, rundir, duration.total_seconds()),
        )
        logging.info(f"{id_} processed.")
    except Exception:
        logging.error(f"{id_} not processed.")
        pass

con.commit()
con.close()
