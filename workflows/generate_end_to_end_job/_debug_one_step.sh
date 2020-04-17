image=us.gcr.io/vcm-ml/prognostic_run:$(git rev-parse HEAD)
input_url=gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/
one_step_yaml=workflows/one_step_jobs/deep-conv-off-fv3atm.yml
output_url=gs://vcm-ml-data/testing-noah/one-step-jobs-$(date +%F)


python workflows/one_step_jobs/orchestrate_submit_jobs.py --n-steps 10 --config-version v0.3\
                                  $input_url $one_step_yaml $image \
                                  $output_url \
				  2>&1 | tee  one-step-$(date +%F).log
