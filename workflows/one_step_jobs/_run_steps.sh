workdir=$(pwd)
 src=gs://vcm-ml-data/orchestration-testing/test-andrep/coarsen_restarts_source-resolution_384_target-resolution_48/
 output=gs://vcm-ml-data/testing-noah/one-step
 image=us.gcr.io/vcm-ml/prognostic_run:v0.1.0-a1
 yaml=$PWD/deep-conv-off.yml

 gsutil -m rm -r $output > /dev/null
 (
    cd ../../
    python $workdir/orchestrate_submit_jobs.py \
        $src $output $yaml $image -o  \
	--config-version v0.3
 )

