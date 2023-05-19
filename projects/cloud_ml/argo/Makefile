SHELL := /bin/bash

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

.PHONY: nudge_to_fine_run_v7
nudge_to_fine_run_v7: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v7 17

.PHONY: extend_nudge_to_fine_run_v6
extend_nudge_to_fine_run_v6: deploy
	cd nudge-to-fine-run && ./extend.sh \
	gs://vcm-ml-experiments/cloud-ml/2023-01-19/cloud-ml-training-data-nudge-to-fine-v6/fv3gfs_run

.PHONY: nudge_to_pire_run
nudge_to_pire_run: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-pire 1

.PHONY: batchable_cloud_state_dataset_v7
batchable_cloud_state_dataset_v7:
	cd ../scripts && python batchable-cloud-state-dataset.py batchable-cloud-state-v7.yaml

.PHONY: batchable_cloud_state_dataset_v8
batchable_cloud_state_dataset_v8:
	cd ../scripts && python batchable-cloud-state-dataset.py batchable-cloud-state-v8.yaml

.PHONY: batchable_cloud_state_dataset_v9
batchable_cloud_state_dataset_v9:
	cd ../scripts && python batchable-cloud-state-dataset.py batchable-cloud-state-v9.yaml

.PHONY: fine_coarse_cloud_state_dataset_v6
fine_coarse_cloud_state_dataset_v6:
	cd ../scripts && python fine-coarse-cloud-state-dataset.py fine-coarse-cloud-state-v6.yaml

.PHONY: fine_coarse_cloud_state_dataset_v7
fine_coarse_cloud_state_dataset_v7:
	cd ../scripts && python fine-coarse-cloud-state-dataset.py fine-coarse-cloud-state-v7.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v6
offline_ml_cloud_predictions_dataset_v6:
	cd ../scripts && python cloud-ml-rf-offline-predictions.py offline-predictions-v6.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v7
offline_ml_cloud_predictions_dataset_v7:
	cd ../scripts && python cloud-ml-rf-offline-predictions.py offline-predictions-v7.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v8
offline_ml_cloud_predictions_dataset_v8:
	cd ../scripts && python cloud-ml-rf-offline-predictions.py offline-predictions-v8.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v9
offline_ml_cloud_predictions_dataset_v9:
	cd ../scripts && python cloud-ml-rf-offline-predictions.py offline-predictions-v9.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v10
offline_ml_cloud_predictions_dataset_v10:
	cd ../scripts && python cloud-ml-rf-offline-predictions.py offline-predictions-v10.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v11
offline_ml_cloud_predictions_dataset_v11:
	cd ../scripts && python cloud-ml-rf-offline-predictions.py offline-predictions-v11.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v12
offline_ml_cloud_predictions_dataset_v12:
	cd ../scripts && python cloud-ml-dense-offline-predictions.py offline-predictions-v12.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v13
offline_ml_cloud_predictions_dataset_v13:
	cd ../scripts && python cloud-ml-dense-offline-predictions.py offline-predictions-v13.yaml

.PHONY: offline_ml_cloud_predictions_dataset_v14
offline_ml_cloud_predictions_dataset_v14:
	cd ../scripts && python cloud-ml-dense-offline-predictions.py offline-predictions-v14.yaml

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

.PHONY: prescribed_cc_cloud_decorr_overlap_run
prescribed_cc_cloud_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-cloud-decorr

.PHONY: prescribed_cc_cloud_ccnorm_max_random_overlap_run
prescribed_cc_cloud_ccnorm_max_random_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-cloud-cc-max-random

.PHONY: prescribed_cc_cloud_max_random_overlap_run
prescribed_cc_cloud_max_random_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-cloud-max-random

.PHONY: prescribed_cc_cloud_ccnorm_ran_overlap_run
prescribed_cc_cloud_ccnorm_ran_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-cloud-cc-ran-ovr

.PHONY: prescribed_cc_cloud_ran_overlap_run
prescribed_cc_cloud_ran_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-cloud-ran-ovr

.PHONY: prescribed_pire_cloud_ccnorm_decorr_overlap_run
prescribed_pire_cloud_ccnorm_decorr_overlap_run: deploy
	cd radiation-port-run && ./run.sh prescribed-pire-cloud-cc-decorr

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

.PHONY: prescribed_cc_rf_ml_gridcell_run
prescribed_cc_rf_ml_gridcell_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell

.PHONY: prescribed_cc_rf_ml_gridcell_cond_01_run
prescribed_cc_rf_ml_gridcell_cond_01_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-cond01

.PHONY: prescribed_cc_rf_ml_gridcell_cond_03_run
prescribed_cc_rf_ml_gridcell_cond_03_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-cond03

.PHONY: prescribed_cc_rf_ml_gridcell_cond_05_run
prescribed_cc_rf_ml_gridcell_cond_05_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-cond05

.PHONY: prescribed_cc_rf_ml_gridcell_cond_08_run
prescribed_cc_rf_ml_gridcell_cond_08_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-cond08

.PHONY: prescribed_cc_rf_ml_gridcell_cond_10_run
prescribed_cc_rf_ml_gridcell_cond_10_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-cond10

.PHONY: prescribed_cc_rf_ml_gridcell_cond_15_run
prescribed_cc_rf_ml_gridcell_cond_15_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-cond15

.PHONY: prescribed_cc_rf_ml_gridcell_cond_25_run
prescribed_cc_rf_ml_gridcell_cond_25_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-cond25

.PHONY: prescribed_cc_rf_ml_gridcell_no_ice_run
prescribed_cc_rf_ml_gridcell_no_ice_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-gridcell-no-ice

.PHONY: prescribed_cc_rf_ml_incloud_run
prescribed_cc_rf_ml_incloud_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud

.PHONY: prescribed_cc_rf_ml_incloud_cond_01_run
prescribed_cc_rf_ml_incloud_cond_01_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud-cond01

.PHONY: prescribed_cc_rf_ml_incloud_cond_03_run
prescribed_cc_rf_ml_incloud_cond_03_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud-cond03

.PHONY: prescribed_cc_rf_ml_incloud_cond_05_run
prescribed_cc_rf_ml_incloud_cond_05_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud-cond05

.PHONY: prescribed_cc_rf_ml_incloud_cond_08_run
prescribed_cc_rf_ml_incloud_cond_08_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud-cond08

.PHONY: prescribed_cc_rf_ml_incloud_cond_10_run
prescribed_cc_rf_ml_incloud_cond_10_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud-cond10

.PHONY: prescribed_cc_rf_ml_incloud_cond_15_run
prescribed_cc_rf_ml_incloud_cond_15_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud-cond15

.PHONY: prescribed_cc_rf_ml_incloud_cond_25_run
prescribed_cc_rf_ml_incloud_cond_25_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-rf-ml-incloud-cond25

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_run
prescribed_cc_nn_ml_gridcell_seed_0_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_cond_01_run
prescribed_cc_nn_ml_gridcell_seed_0_cond_01_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0-cond01

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_cond_03_run
prescribed_cc_nn_ml_gridcell_seed_0_cond_03_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0-cond03

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_cond_05_run
prescribed_cc_nn_ml_gridcell_seed_0_cond_05_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0-cond05

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_cond_08_run
prescribed_cc_nn_ml_gridcell_seed_0_cond_08_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0-cond08

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_cond_10_run
prescribed_cc_nn_ml_gridcell_seed_0_cond_10_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0-cond10

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_cond_15_run
prescribed_cc_nn_ml_gridcell_seed_0_cond_15_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0-cond15

.PHONY: prescribed_cc_nn_ml_gridcell_seed_0_cond_25_run
prescribed_cc_nn_ml_gridcell_seed_0_cond_25_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-seed-0-cond25

.PHONY: prescribed_cc_nn_ml_gridcell_seed_1_run
prescribed_cc_nn_ml_gridcell_seed_1_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-seed-1

.PHONY: prescribed_cc_nn_ml_gridcell_seed_2_run
prescribed_cc_nn_ml_gridcell_seed_2_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-seed-2

.PHONY: prescribed_cc_nn_ml_gridcell_seed_3_run
prescribed_cc_nn_ml_gridcell_seed_3_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-seed-3

.PHONY: prescribed_cc_nn_ml_gridcell_run
prescribed_cc_nn_ml_gridcell_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell

.PHONY: prescribed_cc_nn_ml_gridcell_cond_01_run
prescribed_cc_nn_ml_gridcell_cond_01_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-cond01

.PHONY: prescribed_cc_nn_ml_gridcell_cond_03_run
prescribed_cc_nn_ml_gridcell_cond_03_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-cond03

.PHONY: prescribed_cc_nn_ml_gridcell_cond_05_run
prescribed_cc_nn_ml_gridcell_cond_05_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-cond05

.PHONY: prescribed_cc_nn_ml_gridcell_cond_08_run
prescribed_cc_nn_ml_gridcell_cond_08_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-cond08

.PHONY: prescribed_cc_nn_ml_gridcell_cond_10_run
prescribed_cc_nn_ml_gridcell_cond_10_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-cond10

.PHONY: prescribed_cc_nn_ml_gridcell_cond_15_run
prescribed_cc_nn_ml_gridcell_cond_15_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-cond15

.PHONY: prescribed_cc_nn_ml_gridcell_cond_25_run
prescribed_cc_nn_ml_gridcell_cond_25_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cc-nn-ml-gridcell-cond25

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

.PHONY: train_cloud_ml_dense_local
train_cloud_ml_dense_local: deploy
	cd training && ./run.sh dense-local

.PHONY: train_cloud_ml_dense_local_seed
train_cloud_ml_dense_local_seed: deploy
	cd training && ./run_small.sh dense-local-seed-$(SEED)

.PHONY: train_cloud_ml_dense_local_small_data
train_cloud_ml_dense_local_small_data: deploy
	cd training && ./run_varying_data.sh dense-local-v2 small

.PHONY: train_cloud_ml_dense_local_small_data_v3
train_cloud_ml_dense_local_small_data_v3: deploy
	cd training && ./run_varying_data.sh dense-local-v3 small

.PHONY: train_cloud_ml_dense_local_small_data_train_val
train_cloud_ml_dense_local_small_data_train_val: deploy
	cd training && ./run_varying_data_train_val.sh dense-local-v2 small

.PHONY: train_cloud_ml_dense_local_medium_data
train_cloud_ml_dense_local_medium_data: deploy
	cd training && ./run_varying_data.sh dense-local-v2 medium

.PHONY: train_cloud_ml_dense_local_large_data
train_cloud_ml_dense_local_large_data: deploy
	cd training && ./run_varying_data.sh dense-local-v2 large

.PHONY: train_cloud_ml_dense_local_large_data_train_val
train_cloud_ml_dense_local_large_data_train_val: deploy
	cd training && ./run_varying_data_train_val.sh dense-local-v2 large

.PHONY: train_cloud_ml_dense_local_large_data_v3
train_cloud_ml_dense_local_large_data_v3: deploy
	cd training && ./run_varying_data.sh dense-local-v3 large

.PHONY: train_cloud_ml_dense_local_large_data_train_val_v3
train_cloud_ml_dense_local_large_data_train_val_v3: deploy
	cd training && ./run_varying_data_train_val.sh dense-local-v3 large

.PHONY: train_cloud_ml_dense_local_extra_large_data
train_cloud_ml_dense_local_extra_large_data: deploy
	cd training && ./run_varying_data.sh dense-local-v2 extra-large

.PHONY: train_cloud_ml_dense_local_small_data_v3_seed
train_cloud_ml_dense_local_small_data_v3_seed: deploy
	cd training && ./run_varying_data.sh dense-local-v3-seed-$(SEED) small

.PHONY: train_cloud_ml_dense_local_small_data_train_val_v3
train_cloud_ml_dense_local_small_data_train_val_v3: deploy
	cd training && ./run_varying_data_train_val.sh dense-local-v3-seed-$(SEED) small

.PHONY: train_cloud_ml_dense_local_medium_data_v3_seed
train_cloud_ml_dense_local_medium_data_v3_seed: deploy
	cd training && ./run_varying_data.sh dense-local-v3-seed-$(SEED) medium

.PHONY: train_cloud_ml_dense_local_large_data_v3_seed
train_cloud_ml_dense_local_large_data_v3_seed: deploy
	cd training && ./run_varying_data.sh dense-local-v3-seed-$(SEED) large

.PHONY: train_cloud_ml_dense_local_extra_large_data_v3_seed
train_cloud_ml_dense_local_extra_large_data_v3_seed: deploy
	cd training && ./run_varying_data.sh dense-local-v3-seed-$(SEED) extra-large

.PHONY: train_cloud_ml_rf
train_cloud_ml_rf: deploy
	cd training && ./run.sh rf

.PHONY: train_cloud_ml_rf_no_coarse_ice
train_cloud_ml_rf_no_coarse_ice: deploy
	cd training && ./run.sh rf-no-coarse-ice

.PHONY: train_cloud_ml_rf_incloud
train_cloud_ml_rf_incloud: deploy
	cd training && ./run.sh rf-incloud

.PHONY: train_cloud_ml_rf_incloud_binary1
train_cloud_ml_rf_incloud_binary1: deploy
	cd training && ./run.sh rf-incloud-binary1

.PHONY: train_cloud_ml_rf_incloud_binary10
train_cloud_ml_rf_incloud_binary10: deploy
	cd training && ./run.sh rf-incloud-binary10

clean:
	rm training-sweep/tempfile

sweep_tempfile:
	cd training-sweep && touch "./tempfile" && chmod 744 "./tempfile"

.PHONY: init_training_sweep_dense_lr
init_training_sweep_dense_lr: sweep_tempfile
	cd training-sweep && ./init.sh dense-lr >> "./tempfile"

.PHONY: run_training_sweep_dense_lr
run_training_sweep_dense_lr: deploy_training_sweep init_training_sweep_dense_lr
	cd training-sweep && source "./tempfile"; ./run.sh dense-lr dense-local && rm "./tempfile"

.PHONY: extend_training_sweep_dense_lr
extend_training_sweep_dense_lr: deploy_training_sweep
	cd training-sweep && ./run.sh dense-lr dense-local

.PHONY: init_training_sweep_dense_local_arch
init_training_sweep_dense_local_arch: sweep_tempfile
	cd training-sweep && ./init.sh dense-local-arch >> "./tempfile"

.PHONY: run_training_sweep_dense_local_arch
run_training_sweep_dense_local_arch: deploy_training_sweep init_training_sweep_dense_local_arch
	cd training-sweep && source "./tempfile"; ./run_small.sh dense-local-arch dense-local && rm "./tempfile"

.PHONY: extend_training_sweep_dense_local_arch
extend_training_sweep_dense_local_arch: deploy_training_sweep
	cd training-sweep && ./run_small.sh dense-local-arch dense-local

.PHONY: init_training_sweep_dense_local_depth_width
init_training_sweep_dense_local_depth_width: sweep_tempfile
	cd training-sweep && ./init.sh dense-local-depth-width >> "./tempfile"

.PHONY: run_training_sweep_dense_local_depth_width_small
run_training_sweep_dense_local_depth_width_small: deploy_training_sweep init_training_sweep_dense_local_depth_width
	cd training-sweep && source "./tempfile"; ./run_small.sh dense-local-depth-width dense-local && rm "./tempfile"

.PHONY: extend_training_sweep_dense_local_depth_width_small
extend_training_sweep_dense_local_depth_width_small: deploy_training_sweep
	cd training-sweep && ./run_small.sh dense-local-depth-width dense-local

.PHONY: init_training_sweep_dense_local_lr_schedule
init_training_sweep_dense_local_lr_schedule: sweep_tempfile
	cd training-sweep && ./init.sh dense-local-lr-schedule >> "./tempfile"

.PHONY: run_training_sweep_dense_local_lr_schedule
run_training_sweep_dense_local_lr_schedule: deploy_training_sweep init_training_sweep_dense_local_lr_schedule
	cd training-sweep && source "./tempfile"; ./run.sh dense-local-lr-schedule dense-local-v3 && rm "./tempfile"

.PHONY: extend_training_sweep_dense_local_lr_schedule
extend_training_sweep_dense_local_lr_schedule: deploy_training_sweep
	cd training-sweep && ./run.sh dense-local-lr-schedule dense-local-v3

.PHONY: deploy_training_sweep
deploy_training_sweep: kustomize
	./kustomize build training-sweep/templates | kubectl apply -f -

fine_restarts_to_zarr:
	kubectl apply -f fine-restarts-to-zarr/pod.yaml

.PHONY: deploy
deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0