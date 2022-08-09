nudge_to_fine_run: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine

nudge_to_fine_run_v2: deploy
	cd nudge-to-fine-run && ./run.sh nudge-to-fine-v2

deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	./install_kustomize.sh 3.10.0