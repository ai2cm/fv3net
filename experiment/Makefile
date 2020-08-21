all: submit_extra_vars_long


submit_physicson_hybrid: config_physicson.json templates
	argo submit argo.yaml -f $<

submit: config.json templates
	argo submit argo.yaml -f config.json

submit_extra_vars: config_solin.json templates
	argo submit train_prog_run.yaml -f $<
    
submit_extra_vars_long: prog_run_template
	argo submit prog_run_long.yaml -f config_solin_long.yaml
    
%.json: %.yaml time_configmap.yaml
	bash prepare_config.sh $< $@

templates:
	kubectl apply -f workflow-templates/
    
prog_run_template:
	kubectl apply -f workflow-templates/prognostic-run.yaml
