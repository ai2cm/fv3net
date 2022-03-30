import sqlite3
import wandb
import json
import logging
import datetime

logging.basicConfig(level=logging.INFO)


def insert_run(cur, job):
    summary = {
        key: val
        for key, val in job.summary.items()
        if isinstance(val, (float, str, int))
    }
    cur.execute(
        """REPLACE INTO
        runs(id, state, name, config, job_type, summary, created_at, tags, group_)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job.id,
            job.state,
            job.name,
            job.json_config,
            job.job_type,
            json.dumps(summary),
            job.created_at,
            json.dumps(job.tags),
            job.group,
        ),
    )


def get_last_time():
    cur = con.cursor()
    ans = cur.execute("select max(updated) from history")
    ((time,),) = ans
    return time or ""


con = sqlite3.connect("example.db")
cur = con.cursor()

api = wandb.Api()
last_update = get_last_time()
query_time = datetime.datetime.utcnow().isoformat()
jobs = api.runs(
    "ai2cm/microphysics-emulation",
    filters={"$or": [{"created_at": {"$gt": last_update}}, {"state": "running"}]},
)

for k, job in enumerate(jobs):
    try:
        logging.info(f"Processing {job.id}")
        insert_run(cur, job)
    except Exception as e:
        logging.warn(f"Failed to import {job}. Raised {e}.")
        pass

# Save (commit) the changes
cur2 = con.cursor()
cur2.execute("INSERT into history VALUES (?)", (query_time,))
con.commit()


# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
con.close()
