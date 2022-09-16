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

.PHONY: port_validation_run
port_validation_run: deploy
	cd radiation-port-run && ./run.sh port-validation

.PHONY: prescribed_cloud_run
prescribed_cloud_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cloud

.PHONY: extend_prescribed_cloud_run
extend_prescribed_cloud_run: deploy
	cd radiation-port-run && ./extend.sh \
	gs://vcm-ml-experiments/cloud-ml/2022-09-14/nudge-to-fine-radiation-port-prescribed-cloud/fv3gfs_run

.PHONY: prescribed_state_run
prescribed_state_run: deploy
	cd radiation-port-run && ./run.sh prescribed-state

.PHONY: deploy
deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0