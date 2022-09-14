nudge_to_fine_run: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine

nudge_to_fine_run_v2: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v2

nudge_to_fine_run_v3: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v3

nudge_to_fine_run_v4: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v4

extend_nudge_to_fine_run_v4: deploy
	cd nudge-to-fine-run && ./extend.sh \
	gs://vcm-ml-experiments/cloud-ml/2022-08-23/cloud-ml-training-data-nudge-to-fine-v4/fv3gfs_run

nudge_to_fine_run_v5: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v5

prescribed_cloud_run: deploy
	cd radiation-port-run && ./run.sh prescribed-cloud

deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0