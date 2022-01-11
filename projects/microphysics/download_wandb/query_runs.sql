SELECT train.model,  train.created_at, prognostic_runs.online
FROM  prognostic_runs
INNER JOIN train ON prognostic_runs.model GLOB train.model || '*'
limit 5
;
