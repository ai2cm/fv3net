submit:
	gsutil cp diag_table gs://vcm-ml-experiments/diag_tables/year-long-baseline-run/v1.0/diag_table
	kubectl apply -k fv3net/workflows/argo
	argo submit --from workflowtemplate/prognostic-run \
		-f config.yaml