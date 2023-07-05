.PHONY: surface_reference_zarr
surface_reference_zarr:
	cd scripts && python ./surface_reference_zarr.py

.PHONY: fine_restarts_to_zarr
fine_restarts_to_zarr:
	cd argo/fine-restarts-to-zarr && kubectl apply -f pod.yaml

.PHONY: nudge_to_fine_baseline_run
nudge_to_fine_baseline_run: deploy
	cd argo/prognostic-run && ./run.sh nudge-to-fine-baseline 39

.PHONY: extend_prognostic_run
extend_prognostic_run: deploy
	cd argo/prognostic-run && ./extend.sh $(URL) $(SEGMENTS)

.PHONY: training_data_zarr
training_data_zarr:
	cd scripts && python ./training_data_zarr.py training_data_zarr_config.yaml

.PHONY: prescribed_cloud_cc_decorr_run
prescribed_cloud_cc_decorr_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-cc-decorr 39

.PHONY: prescribed_cloud_cc_max_random_run
prescribed_cloud_cc_max_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-cc-max-random 39

.PHONY: prescribed_cloud_cc_random_run
prescribed_cloud_cc_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-cc-random 39

.PHONY: prescribed_cloud_decorr_run
prescribed_cloud_decorr_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-decorr 39

.PHONY: prescribed_cloud_max_random_run
prescribed_cloud_max_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-max-random 39

.PHONY: prescribed_cloud_random_run
prescribed_cloud_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-random 39

.PHONY: train_cloud_ml_dense_seed
train_cloud_ml_dense_seed: deploy
	cd argo/training && ./run_random_seed.sh dense 4

.PHONY: upload_squashed_models
upload_squashed_models:
	cd scripts/upload-squashed-models && ./upload.sh 4 gs://vcm-ml-experiments/cloud-ml/2023-07-04/train-cloud-ml-dense

.PHONY: prescribed_cloud_dense_seed_squash_threshold
prescribed_cloud_dense_seed_squash_threshold: deploy
	cd argo/prognostic-run && ./run_seed_squash_threshold.sh prescribed-cloud-dense 4 39

.PHONY: deploy
deploy: argo/kustomize
	cd argo && ./kustomize build . | kubectl apply -f -

argo/kustomize:
	cd argo && ./install_kustomize.sh 3.10.0