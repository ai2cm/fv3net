nudge_to_fine_baseline_run: deploy
	cd argo/prognostic-run && ./run.sh nudge-to-fine-baseline

deploy: argo/kustomize
	cd argo && ./kustomize build . | kubectl apply -f -

argo/kustomize:
	cd argo && ./install_kustomize.sh 3.10.0