.open example.db
.mode csv
.header on


select *
FROM (
    SELECT group_ as "group"
        -- , group_concat(job_type)
        , max(json_each.value = 'experiment/longer-runs') as match
        , max(model) model
        , max(gscond_only) gscond_only
        , max(online) as online
        , max(json_extract(summary, '$.duration_seconds') / 86400.0) as days
        , max(json_extract(summary, '$.column_skill/surface_precipitation')) as prec_skill
    FROM json_each(runs.tags), (
        SELECT *
        , IFNULL(
            json_extract(config, '$.config.value.zhao_carr_emulation.model.path')
        ,   json_extract(config, '$.config.value.zhao_carr_emulation.gscond.path')) model
        ,   IFNULL(json_extract(config, '$.config.value.namelist.gfs_physics_nml.emulate_gscond_only'), 0) gscond_only
        , json_extract(config, '$.config.value.namelist.gfs_physics_nml.emulate_zc_microphysics') as online
        FROM runs
        ) runs
    GROUP BY group_
    )
WHERE match AND days AND online
;