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

.PHONY: nudge_to_fine_run_v6
nudge_to_fine_run_v6: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v6

.PHONY: batchable_cloud_state_dataset_v6
batchable_cloud_state_dataset_v6:
	cd ../scripts && python batchable-cloud-state-dataset.py batchable-v6.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v6
offline_ml_cloud_predictions_dataset_v6:
	cd ../scripts && python cloud-ml-rf-offline-predictions.py offline-predictions-v6.yaml

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

.PHONY: baseline_cc_ccnorm_decorr_overlap_port_run
baseline_cc_ccnorm_decorr_overlap_port_run: deploy
	cd radiation-port-run && ./run.sh baseline-cc-ccnorm-decorr-port

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

.PHONY: port_validation_cc_ccnorm_decorr_overlap_run
port_validation_cc_ccnorm_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh port-validation-cc-ccnorm-decorr

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

.PHONY: prescribed_cc_cloud_ccnorm_decorr_overlap_run
prescribed_cc_cloud_ccnorm_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-cloud-cc-decorr

.PHONY: prescribed_ml_cloud_run
prescribed_ml_cloud_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud

.PHONY: prescribed_ml_cloud_cc_run
prescribed_ml_cloud_cc_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cc

.PHONY: prescribed_ml_cloud_decorr_run
prescribed_ml_cloud_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-decorr

.PHONY: prescribed_ml_cloud_cc_decorr_run
prescribed_ml_cloud_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cc-decorr

.PHONY: prescribed_ml_cloud_cond01_run
prescribed_ml_cloud_cond01_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cond01

.PHONY: prescribed_ml_cloud_cond03_run
prescribed_ml_cloud_cond03_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cond03

.PHONY: prescribed_ml_cloud_cond06_run
prescribed_ml_cloud_cond06_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cond06

.PHONY: prescribed_ml_cloud_cond10_run
prescribed_ml_cloud_cond10_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cond10

.PHONY: prescribed_ml_cloud_cond15_run
prescribed_ml_cloud_cond15_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cond15

.PHONY: prescribed_ml_cloud_cond25_run
prescribed_ml_cloud_cond25_run: deploy
	cd radiation-port-run && ./run.sh prescribed-ml-cloud-cond25

.PHONY: prescribed_rf_ml_cloud_cc_decorr_run
prescribed_rf_ml_cloud_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cloud-cc-decorr

.PHONY: prescribed_rf_cloud_cond01_run
prescribed_rf_cloud_cond01_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cloud-cond01

.PHONY: prescribed_rf_cloud_cond03_run
prescribed_rf_cloud_cond03_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cloud-cond03

.PHONY: prescribed_rf_cloud_cond06_run
prescribed_rf_cloud_cond06_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cloud-cond06

.PHONY: prescribed_rf_cloud_cond10_run
prescribed_rf_cloud_cond10_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cloud-cond10

.PHONY: prescribed_rf_cloud_cond15_run
prescribed_rf_cloud_cond15_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cloud-cond15

.PHONY: prescribed_rf_ml_incloud_cc_decorr_run
prescribed_rf_ml_incloud_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-incloud-cc-decorr

.PHONY: prescribed_rf_ml_cond_01_cc_decorr_run
prescribed_rf_ml_cond_01_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cond01-cc-decorr

.PHONY: prescribed_rf_ml_cond_03_cc_decorr_run
prescribed_rf_ml_cond_03_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cond03-cc-decorr

.PHONY: prescribed_rf_ml_cond_06_cc_decorr_run
prescribed_rf_ml_cond_06_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cond06-cc-decorr

.PHONY: prescribed_rf_ml_cond_10_cc_decorr_run
prescribed_rf_ml_cond_10_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cond10-cc-decorr

.PHONY: prescribed_rf_ml_cond_15_cc_decorr_run
prescribed_rf_ml_cond_15_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cond15-cc-decorr

.PHONY: prescribed_rf_ml_cond_level_cc_decorr_run
prescribed_rf_ml_cond_level_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-cond-level-cc-decorr

.PHONY: prescribed_rf_ml_quantile_cc_decorr_run
prescribed_rf_ml_quantile_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-quantile-cc-decorr

.PHONY: prescribed_rf_ml_quantilein_cc_decorr_run
prescribed_rf_ml_quantilein_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-quantilein-cc-decorr

.PHONY: prescribed_rf_ml_quantile01_cc_decorr_run
prescribed_rf_ml_quantile01_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-rf-quantile01-cc-decorr

.PHONY: prescribed_cc_rf_ml_cond_10_cc_decorr_run
prescribed_cc_rf_ml_cond_10_cc_decorr_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-cond10-cc-decorr

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

.PHONY: train_cloud_ml_dense
train_cloud_ml_dense: deploy
	cd training && ./run.sh dense

.PHONY: train_cloud_ml_rf
train_cloud_ml_rf: deploy
	cd training && ./run.sh rf

.PHONY: train_cloud_ml_rf_incloud
train_cloud_ml_rf_incloud: deploy
	cd training && ./run.sh rf-incloud

.PHONY: train_cloud_ml_rf_incloud_binary1
train_cloud_ml_rf_incloud_binary1: deploy
	cd training && ./run.sh rf-incloud-binary1

.PHONY: train_cloud_ml_rf_incloud_binary10
train_cloud_ml_rf_incloud_binary10: deploy
	cd training && ./run.sh rf-incloud-binary10

fine_restarts_to_zarr:
	kubectl apply -f fine-restarts-to-zarr/pod.yaml

.PHONY: deploy
deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0