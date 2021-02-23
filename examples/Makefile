
nudge_to_fine_run: deploy
	cd nudge-to-fine-run; ./run.sh

nudge_to_obs_run: deploy
	cd nudge-to-obs-run; ./run.sh

prognostic_run: deploy
	cd prognostic-run; ./run.sh

prognostic_run_report: deploy
	cd prognostic-run-report && ./run.sh

train_evaluate_prognostic_run: deploy
	cd train-evaluate-prognostic-run; ./run.sh


deploy: kustomize
	./kustomize build . | kubectl apply -f -

kustomize:
	../install_kustomize.sh 3.10.0

.PHONY: nudge_to_fine_run nudge_to_obs_run prognostic_run train_evaluate_prognostic_run deploy prognostic_run_report
