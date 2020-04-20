set -e 
set -x
OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/prognostic_run
TRAINED_MODEL=gs://vcm-ml-data/testing-noah/ceb320ffa48b8f8507b351387ffa47d1b05cd402/sklearn_train/
IC=20160801.001500
image=us.gcr.io/vcm-ml/prognostic_run:35347c79cf47a77f54639632c5fadb698da7b2f1
ONE_STEP_RUNS=gs://vcm-ml-data/testing-noah/one-step-jobs-2020-04-13

python ../../workflows/prognostic_c48_run/orchestrate_submit_job.py \
	--prog_config_yml prognostic_run.yml  \
	--model_url  $TRAINED_MODEL \
	$ONE_STEP_RUNS \
	$IC \
	$image \
	$OUTPUT
