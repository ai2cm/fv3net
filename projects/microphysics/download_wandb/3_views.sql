DROP VIEW IF EXISTS prognostic_runs;
CREATE VIEW prognostic_runs AS
SELECT *,
    json_extract(config, '$.env.value.TF_MODEL_PATH') as model
    , json_extract(config, '$.config.value.namelist.gfs_physics_nml.emulate_zc_microphysics') as online
    , json_extract(config, '$.rundir.value') as location
    , json_extract(piggy.summary, '$.duration_seconds') / 86400.0 as days
    , SQRT(json_extract(train.summary, "$.val_specific_humidity_after_precpd_loss")) * 86400 / 900 as qv_rmse_kg_kg_day
    , json_extract(eval.summary, '$.drifts/cloud_water_mixing_ratio/5day') as cloud_drift_5d
    , json_extract(eval.summary, '$.drifts/cloud_water_mixing_ratio/10day') as cloud_drift_10d
    , json_extract(eval.summary, '$.drifts/air_temperature/5day') as temp_drift_5d
    , json_extract(piggy.summary, '$.column_skill/surface_precipitation') as prec_skill
FROM runs
where job_type ='prognostic_run';

DROP VIEW IF EXISTS train;
CREATE VIEW train AS
SELECT *,
    json_extract(config, '$.out_url.value') as model
FROM runs
where job_type ='train';
