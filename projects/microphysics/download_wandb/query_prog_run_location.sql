.mode csv
.output stats.csv
.header on

SELECT DISTINCT train.name,
    location,
    duration_seconds / 86400.0 as days,
    SQRT(json_extract(train.summary, "$.val_specific_humidity_after_precpd_loss")) * 86400 / 900 as val_specific_humidity_after_precpd_rmse_kg_kg_day
FROM  prognostic_runs
INNER JOIN train ON prognostic_runs.model GLOB train.model || '*'
INNER JOIN progsummary ON progsummary.run_id = prognostic_runs.id
WHERE online
ORDER BY days DESC
;
