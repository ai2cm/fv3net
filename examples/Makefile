
nudge_to_fine_run: kustomize
	./kustomize build . | kubectl apply -f -
	cd nudge-to-fine-run; ./run.sh

nudge_to_obs_run: kustomize
	./kustomize build . | kubectl apply -f -
	cd nudge-to-obs-run; ./run.sh

prognostic_run: kustomize
	./kustomize build . | kubectl apply -f -
	cd prognostic-run; ./run.sh

train_evaluate_prognostic_run: kustomize
	./kustomize build . | kubectl apply -f -
	cd train-evaluate-prognostic-run; ./run.sh

.PHONY: nudge_to_fine_run nudge_to_obs_run prognostic_run train_evaluate_prognostic_run

kustomize:
	../install_kustomize.sh 3.8.6

