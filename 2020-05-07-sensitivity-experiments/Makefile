EXPERIMENTS = $(shell find 1_experiments -maxdepth 1 -mindepth 1 -type d) 

check_configs:
	kubectl apply -k 0_create_training_data --dry-run -o yaml > /dev/null
	for exp_dir in $(EXPERIMENTS) ; do \
		kubectl apply -k $$exp_dir --dry-run -o yaml ; \
	done ; \
	echo "Kustomize configs are ok."

create_training_data:
	kubectl apply -k 0_create_training_data
	
run_experiments:
	for exp_dir in $(EXPERIMENTS) ; do \
		kubectl apply -k $$exp_dir ; \
	done ;
