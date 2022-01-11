DROP VIEW IF EXISTS prognostic_runs;
CREATE VIEW prognostic_runs AS
SELECT *, 
    json_extract(config, '$.env.value.TF_MODEL_PATH') as model,
    json_extract(config, '$.config.value.namelist.gfs_physics_nml.emulate_zc_microphysics') as online
FROM runs
where job_type ='prognostic_run';

DROP VIEW IF EXISTS train;
CREATE VIEW train AS
SELECT *, 
    json_extract(config, '$.out_url.value') as model
FROM runs
where job_type ='train';
