submit: config.json
	kubectl apply -f workflow-templates/
	argo submit argo.yaml -f config.json


config.json: config.yaml time_configmap.yaml
	bash prepare_config.sh
