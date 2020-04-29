
set -e
set -x

source ../end_to_end_integration/kustomization/input_data.env
PROGNOSTIC_RUN_IMAGE=us.gcr.io/vcm-ml/prognostic_run:$(git rev-parse HEAD)
OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/one_steps
ONE_STEP_TIMES=times.json

python ../../workflows/one_step_jobs/orchestrate_submit_jobs.py \
        --config-version v0.3 \
          $C48_RESTARTS \
        one_step_jobs.yml \
        $PROGNOSTIC_RUN_IMAGE \
        $ONE_STEP_TIMES \
	$OUTPUT
