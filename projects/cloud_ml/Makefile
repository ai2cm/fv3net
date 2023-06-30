.PHONY: surface_reference_zarr
surface_reference_zarr:
	cd scripts && python ./surface_reference_zarr.py

.PHONY: nudge_to_fine_baseline_run
nudge_to_fine_baseline_run: deploy
	cd argo/prognostic-run && ./run.sh nudge-to-fine-baseline

.PHONY: deploy
deploy: argo/kustomize
	cd argo && ./kustomize build . | kubectl apply -f -

argo/kustomize:
	cd argo && ./install_kustomize.sh 3.10.0