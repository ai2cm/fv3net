.PHONY: nudge_to_fine_run
nudge_to_fine_run: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine

.PHONY: nudge_to_fine_run_v2
nudge_to_fine_run_v2: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v2

.PHONY: nudge_to_fine_run_v3
nudge_to_fine_run_v3: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v3

.PHONY: nudge_to_fine_run_v4
nudge_to_fine_run_v4: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v4

.PHONY: extend_nudge_to_fine_run_v4
extend_nudge_to_fine_run_v4: deploy
	cd nudge-to-fine-run && ./extend.sh \
	gs://vcm-ml-experiments/cloud-ml/2022-08-23/cloud-ml-training-data-nudge-to-fine-v4/fv3gfs_run

.PHONY: nudge_to_fine_run_v5
nudge_to_fine_run_v5: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v5

.PHONY: baseline_port_run
baseline_port_run: deploy
	cd radiation-port-run && ./run.sh baseline-port

.PHONY: baseline_ccnorm_port_run
baseline_ccnorm_port_run: deploy
	cd radiation-port-run && ./run.sh baseline-ccnorm-port

.PHONY: baseline_random_overlap_port_run
baseline_random_overlap_port_run: deploy
	cd radiation-port-run && ./run.sh baseline-random-overlap-port

.PHONY: baseline_decorr_overlap_port_run
baseline_decorr_overlap_port_run: deploy
	cd radiation-port-run && ./run.sh baseline-decorr-port

.PHONY: baseline_ccnorm_random_overlap_port_run
baseline_ccnorm_random_overlap_port_run: deploy
	cd radiation-port-run && ./run.sh baseline-cc-ran-ovr-port

.PHONY: baseline_ccnorm_decorr_overlap_port_run
baseline_ccnorm_decorr_overlap_port_run: deploy
	cd radiation-port-run && ./run.sh baseline-cc-decorr-port

.PHONY: port_validation_run
port_validation_run: deploy
	cd radiation-port-run && ./run.sh port-validation

.PHONY: port_validation_run_no_mc
port_validation_run_no_mc: deploy
	cd radiation-port-run && ./run.sh port-validation-no-mc

.PHONY: port_validation_random_overlap_run
port_validation_random_overlap_run: deploy
	cd radiation-port-run && ./run.sh port-validation-ran-ovr

.PHONY: port_validation_decorr_overlap_run
port_validation_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh port-validation-decorr

.PHONY: port_validation_ccnorm_random_overlap_run
port_validation_ccnorm_random_overlap_run: deploy
	cd radiation-port-run && ./run.sh port-validation-cc-ran-ovr

.PHONY: port_validation_ccnorm_decorr_overlap_run
port_validation_ccnorm_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh port-validation-cc-decorr

.PHONY: prescribed_cloud_run
prescribed_cloud_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cloud

.PHONY: prescribed_cloud_random_overlap_run
prescribed_cloud_random_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cloud-ran-ovr

.PHONY: prescribed_cloud_decorr_overlap_run
prescribed_cloud_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cloud-decorr

.PHONY: prescribed_cloud_ccnorm_random_overlap_run
prescribed_cloud_ccnorm_random_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cloud-cc-ran-ovr

.PHONY: prescribed_cloud_ccnorm_decorr_overlap_run
prescribed_cloud_ccnorm_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cloud-cc-decorr

.PHONY: prescribed_half_cloud_run
prescribed_half_cloud_run: deploy
	cd radiation-port-run && ./run.sh prescribed-half-cloud

.PHONY: extend_prescribed_cloud_run
extend_prescribed_cloud_run: deploy
	cd radiation-port-run && ./extend.sh \
	gs://vcm-ml-experiments/cloud-ml/2022-09-14/nudge-to-fine-radiation-port-prescribed-cloud/fv3gfs_run

.PHONY: prescribed_state_run
prescribed_state_run: deploy
	cd radiation-port-run && ./run.sh prescribed-state

.PHONY: baseline_run_max_ran_rad
baseline_run_max_ran_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh baseline-max-ran-rad

.PHONY: baseline_run_max_ran_deterministic_rad
baseline_run_max_ran_deterministic_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh baseline-max-ran-deterministic-rad

.PHONY: baseline_run_ran_ovr_rad
baseline_run_ran_ovr_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh baseline-ran-ovr-rad

.PHONY: baseline_run_max_ovr_rad
baseline_run_max_ovr_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh baseline-max-ovr-rad

.PHONY: baseline_run_decorr_rad
baseline_run_decorr_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh baseline-decorr-rad

.PHONY: baseline_yearlong_run
baseline_yearlong_run: deploy
	cd baseline && ./run.sh baseline-yearlong

.PHONY: baseline_yearlong_run_ccnorm
baseline_yearlong_run_ccnorm: deploy
	cd baseline && ./run.sh baseline-yearlong-ccnorm

.PHONY: nudge_to_fine_run_max_ran_rad
nudge_to_fine_run_max_ran_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh nudge-to-fine-max-ran-rad

.PHONY: nudge_to_fine_run_max_ran_deterministic_rad
nudge_to_fine_run_max_ran_deterministic_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh nudge-to-fine-max-ran-determ-rad

.PHONY: nudge_to_fine_run_ran_ovr_rad
nudge_to_fine_run_ran_ovr_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh nudge-to-fine-ran-ovr-rad

.PHONY: nudge_to_fine_run_max_ovr_rad
nudge_to_fine_run_max_ovr_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh nudge-to-fine-max-ovr-rad

.PHONY: nudge_to_fine_run_decorr_rad
nudge_to_fine_run_decorr_rad: deploy
	cd nudge-to-fine-run/rad-config && ./run.sh nudge-to-fine-decorr-rad

.PHONY: rad_overlap_report
rad_overlap_report: deploy
	cd reports && ./submit.sh rad-overlap

.PHONY: train_cloud_ml
train_cloud_ml: deploy
	cd training && ./run.sh

.PHONY: deploy
deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0