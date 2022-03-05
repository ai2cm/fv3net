CREATE TABLE runs (
    id text not null primary key,
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

