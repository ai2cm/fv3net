all: submit_extra_vars_long

submit_extra_vars_long: prog_run_template
	argo submit prog_run_long.yaml -f config_solin_long.yaml
    
prog_run_template:
	kubectl apply -f workflow-templates/prognostic-run.yaml
