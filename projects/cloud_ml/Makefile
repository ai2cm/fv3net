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

.PHONY: deploy
deploy: argo/kustomize
	cd argo && ./kustomize build . | kubectl apply -f -

argo/kustomize:
	cd argo && ./install_kustomize.sh 3.10.0