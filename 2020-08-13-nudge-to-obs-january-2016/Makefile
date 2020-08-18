submit_nudge_to_obs:
	kubectl apply -f fv3net/workflows/argo/nudge_to_obs/workflow.yaml
	argo submit --from workflowtemplate/nudge-to-obs \
		-f nudge_config.yaml