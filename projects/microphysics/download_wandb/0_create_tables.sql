CREATE TABLE runs (
    id integer not null primary key,
    wandb_id text,
    name text,
    config text,
    summary text,
    job_type text,
    created_at text
);

CREATE TABLE progsummary (
    run_id integer not null unique,
    location text,
    duration_seconds integer,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
