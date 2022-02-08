import sqlite3
import wandb
import json
import logging

logging.basicConfig(level=logging.INFO)


def insert_run(cur, job):
    summary = {
        key: val
        for key, val in job.summary.items()
        if isinstance(val, (float, str, int))
    }
    cur.execute(
        """INSERT INTO runs(wandb_id, name, config, job_type, summary, created_at, tags)
    VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            job.id,
            job.name,
            job.json_config,
            job.job_type,
            json.dumps(summary),
            job.created_at,
            json.dumps(job.tags),
        ),
    )


con = sqlite3.connect("example.db")
cur = con.cursor()

api = wandb.Api()
jobs = api.runs("ai2cm/microphysics-emulation")

for k, job in enumerate(jobs):
    try:
        insert_run(cur, job)
    except sqlite3.OperationalError:
        logging.warn(f"Failed to import {job}")
        pass

# Save (commit) the changes
con.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
con.close()
