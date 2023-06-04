.PHONY: surface_reference_zarr
surface_reference_zarr:
	cd scripts && python ./surface_reference_zarr.py

.PHONY: nudge_to_fine_baseline_run
nudge_to_fine_baseline_run: deploy
	cd argo/prognostic-run && ./run.sh nudge-to-fine-baseline

.PHONY: training_data_zarr
training_data_zarr:
	cd scripts && python ./training_data_zarr.py training_data_config.yaml

.PHONY: prescribed_cloud_cc_decorr_run
prescribed_cloud_cc_decorr_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-cc-decorr

.PHONY: prescribed_cloud_cc_max_random_run
prescribed_cloud_cc_max_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-cc-max-random

.PHONY: prescribed_cloud_cc_random_run
prescribed_cloud_cc_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-cc-random

.PHONY: prescribed_cloud_decorr_run
prescribed_cloud_decorr_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-decorr

.PHONY: prescribed_cloud_max_random_run
prescribed_cloud_max_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-max-random

.PHONY: prescribed_cloud_random_run
prescribed_cloud_random_run: deploy
	cd argo/prognostic-run && ./run.sh prescribed-cloud-random

.PHONY: prescribed_sub_fine_cc_decorr_run
prescribed_sub_fine_cc_decorr_run: deploy
	cd argo/prognostic-run && ./run_random_seed.sh prescribed-sub-fine-cc-decorr

.PHONY: prescribed_sub_fine_cc_max_random_run
prescribed_sub_fine_cc_max_random_run: deploy
	cd argo/prognostic-run && ./run_random_seed.sh prescribed-sub-fine-cc-max-random

.PHONY: deploy
deploy: argo/kustomize
	cd argo && ./kustomize build . | kubectl apply -f -

argo/kustomize:
	cd argo && ./install_kustomize.sh 3.10.0