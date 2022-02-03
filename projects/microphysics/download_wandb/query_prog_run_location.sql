-- .mode csv
-- .output stats.csv
.mode column
.header on

SELECT DISTINCT train.name,
    location
    , duration_seconds / 86400.0 as days
    , SQRT(json_extract(train.summary, "$.val_specific_humidity_after_precpd_loss")) * 86400 / 900 as qv_rmse_kg_kg_day
    , json_extract(eval.summary, '$.drifts/cloud_water_mixing_ratio/5day') as cloud_drift_5d
    , json_extract(eval.summary, '$.drifts/cloud_water_mixing_ratio/10day') as cloud_drift_10d
    , json_extract(eval.summary, '$.drifts/air_temperature/5day') as temp_drift_5d
    , json_extract(piggy.summary, '$.column_skill/surface_precipitation') as prec_skill
FROM  prognostic_runs
INNER JOIN train ON prognostic_runs.model GLOB train.model || '*'
INNER JOIN progsummary ON progsummary.run_id = prognostic_runs.id
INNER JOIN (
    SELECT *
        , json_extract(config, '$.run.value') as rundir
    FROM runs
    WHERE job_type='piggy-back'
) piggy ON location = rundir || '/'
INNER JOIN (
    SELECT *
        , json_extract(config, '$.run.value') as zarrinrundir
    FROM runs
    WHERE job_type='prognostic_evaluation'
) eval ON zarrinrundir GLOB location || '*'
WHERE online
GROUP BY location
ORDER BY duration_seconds DESC, ABS(cloud_drift_5d) ASC
;
