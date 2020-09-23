PWD=$(shell pwd)
OUTPUT_TIMES = $(PWD)/training-simulations/output-times.json
TRAINING_TIMES = $(PWD)/ml-experiments/training-times.json
TESTING_TIMES = $(PWD)/ml-experiments/testing-times.json


submit_physics_on_nudge_to_fine_training_simulation:
	gsutil -m cp $(PWD)/training-simulations/diag_table gs://vcm-ml-experiments/diag_tables/nudge_to_fine_2h/v1.0/diag_table
	kubectl apply -k fv3net/workflows/argo/
	$(PWD)/training-simulations/run.sh \
		$(PWD)/training-simulations/nudging-config-physics-on.yml \
		$(OUTPUT_TIMES) \
		physics-on-nudge-to-fine


submit_clouds_off_nudge_to_fine_training_simulation:
	gsutil -m cp $(PWD)/training-simulations/diag_table gs://vcm-ml-experiments/diag_tables/nudge_to_fine_2h/v1.0/diag_table
	kubectl apply -k fv3net/workflows/argo/
	$(PWD)/training-simulations/run.sh \
		$(PWD)/training-simulations/nudging-config-clouds-off.yml \
		$(OUTPUT_TIMES) \
		clouds-off-nudge-to-fine


submit_train_diags_prog_hybrid_fortran_pQ_no_nudging_tendencies:
	kubectl apply -k fv3net/workflows/argo/
	$(PWD)/ml-experiments/run.sh \
		$(PWD)/ml-experiments/fortran-pQ-no-nudging-tendencies/training.yml \
		$(PWD)/ml-experiments/fortran-pQ-no-nudging-tendencies/prognostic-run.yml \
		$(TRAINING_TIMES) \
		$(TESTING_TIMES) \
		hybrid-fortran-pQ-no-nudging-tendencies


submit_train_diags_prog_hybrid_fortran_pQ_nudge_to_fine_tendencies:
	kubectl apply -k fv3net/workflows/argo/
	$(PWD)/ml-experiments/run.sh \
		$(PWD)/ml-experiments/fortran-pQ-nudge-to-fine-tendencies/training.yml \
		$(PWD)/ml-experiments/fortran-pQ-nudge-to-fine-tendencies/prognostic-run.yml \
		$(TRAINING_TIMES) \
		$(TESTING_TIMES) \
		hybrid-fortran-pQ-nudge-to-fine-tendencies


submit_train_diags_prog_hybrid_fortran_pQ_xshield_nudging_tendency:
	kubectl apply -k fv3net/workflows/argo/
	$(PWD)/ml-experiments/run.sh \
		$(PWD)/ml-experiments/fortran-pQ-xshield-nudging-tendency/training.yml \
		$(PWD)/ml-experiments/fortran-pQ-xshield-nudging-tendency/prognostic-run.yml \
		$(TRAINING_TIMES) \
		$(TESTING_TIMES) \
		hybrid-fortran-pQ-xshield-nudging-tendency


submit_train_diags_prog_hybrid_nudge_to_fine_pQ_no_nudging_tendencies:
	kubectl apply -k fv3net/workflows/argo/
	$(PWD)/ml-experiments/run.sh \
		$(PWD)/ml-experiments/nudge-to-fine-pQ-no-nudging-tendencies/training.yml \
		$(PWD)/ml-experiments/nudge-to-fine-pQ-no-nudging-tendencies/prognostic-run.yml \
		$(TRAINING_TIMES) \
		$(TESTING_TIMES) \
		hybrid-nudge-to-fine-pQ-no-nudging-tendencies
