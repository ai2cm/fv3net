submit_extra_vars: config_solin.json templates
	argo submit train_prog_run.yaml -f $<

submit: config.json templates
	argo submit argo.yaml -f config.json


%.json: %.yaml time_configmap.yaml
	bash prepare_config.sh $< $@


templates:
	kubectl apply -f workflow-templates/
