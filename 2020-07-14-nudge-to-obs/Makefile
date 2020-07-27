
submit_nudge_to_obs:
	kubectl apply -f fv3net/workflows/argo/nudge_to_obs/workflow.yaml
	argo submit --from workflowtemplate/nudge-to-obs \
		-f 0_nudge_to_obs_run/nudge_config.yaml

submit_train_and_prognostic:
	kubectl apply -k 1_train_and_prognostic
