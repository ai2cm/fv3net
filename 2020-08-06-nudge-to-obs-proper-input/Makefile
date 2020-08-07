TRAIN_TIMES = ../2020-07-14-nudge-to-obs/1_train_and_prognostic/training_times.json

submit:
	bash prepare_config.sh config.yaml $(TRAIN_TIMES) > compiled_config.yaml
	kubectl apply -k fv3net/workflows/argo/
	argo submit argo.yaml -f compiled_config.yaml
