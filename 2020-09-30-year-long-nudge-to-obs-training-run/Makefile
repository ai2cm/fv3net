submit_nudge_to_obs:
	gsutil cp diag_table gs://vcm-ml-experiments/diag_tables/nudge_to_obs_5h/v1.0/diag_table
	kubectl apply -k fv3net/workflows/argo
	argo submit --from workflowtemplate/nudge-to-obs -f nudge_config.yaml

copy_gfs_analysis_to_pvc:
	argo submit fv3net/workflows/argo/copy-data-to-read-only-pvc.yaml \
		-p size="200Gi" \
		-p pvc-name="gfs-analysis-read-only" \
		-p source="gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data/*" \
		-p destination="gfs-analysis-T85"