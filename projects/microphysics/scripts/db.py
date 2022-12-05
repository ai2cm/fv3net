import json
import sqlite3
import logging

CREATE_TABLES_SQL = """
CREATE TABLE runs (
    id text not null primary key,
    state text,
    name text,
    config text,
    summary text,
    job_type text,
    tags text,
    created_at text,
    group_ text
);

CREATE UNIQUE INDEX idx_runs_id ON runs (id);

CREATE TABLE progsummary (
    run_id integer not null unique,
    location text,
    duration_seconds integer,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def create_tables(connection):
    connection.executescript(CREATE_TABLES_SQL)


connection = sqlite3.connect(":memory:")
create_tables(connection)


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


def insert_runs(jobs):
    cur = connection.cursor()
    for _, job in enumerate(jobs):
        try:
            logging.info(f"Processing {job.id}")
            insert_run(cur, job)
        except Exception as e:
            logging.warn(f"Failed to import {job}. Raised {e}.")
            pass
    connection.commit()


def query(sql, *args):
    cur = connection.cursor()
    return cur.execute(sql, *args)
