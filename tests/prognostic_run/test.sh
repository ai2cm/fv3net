set -e 
set -x
OUTPUT=gs://vcm-ml-data/testing-noah/$(git rev-parse HEAD)/prognostic_run
TRAINED_MODEL=gs://vcm-ml-data/testing-noah/ceb320ffa48b8f8507b351387ffa47d1b05cd402/sklearn_train/
IC_URL=gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48
IC=20160801.001500
image=us.gcr.io/vcm-ml/prognostic_run:v0.3.2

python ../../workflows/prognostic_c48_run/orchestrate_submit_job.py \
	--model_url  $TRAINED_MODEL \
	$IC_URL \
	$IC \
	$image \
	prognostic_run.yml  \
	$OUTPUT
