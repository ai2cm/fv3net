CREATE TABLE runs (
    id integer not null primary key,
    wandb_id text,
    name text,
    config text,
    summary text,
    job_type text,
    created_at text
);

