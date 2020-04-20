OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/coarsen_diagnostics/

python ../../workflows/prognostic_c48_run/orchestrate_submit_job.py \
	--prog_config_yml prognostic_run.yml  \
	--model_url  gs://vcm-ml-data/test-end-to-end-integration/integration-debug/train_sklearn_model \
	gs://vcm-ml-data/test-end-to-end-integration/integration-debug/one_step_run \
	20160801.001500 \
	us.gcr.io/vcm-ml/prognostic_run:35347c79cf47a77f54639632c5fadb698da7b2f1 \
	$OUTPUT
