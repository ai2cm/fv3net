#!/bin/bash

set -xe

[[ -n $GOOGLE_APPLICATION_CREDENTIALS ]] && gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS

#RUN=gs://vcm-ml-code-testing-data/sample-prognostic-run-output
RUN=gs://vcm-ml-experiments/2021-01-26-c3072-nn/combined-prognostic/seed_0 

random=$(openssl rand --hex 6)
OUTPUT=gs://vcm-ml-scratch/test-prognostic-report/$random

cd workflows/prognostic_run_diags

# compute diagnostics/mterics for a short sample prognostic run
mkdir -p /tmp/$random
prognostic_run_diags save $RUN /tmp/$random/diags.nc
prognostic_run_diags metrics /tmp/$random/diags.nc > /tmp/$random/metrics.json

cp  /tmp/$random/diags.nc /home/AnnaK/fv3net/scratch/2021-02-24-prog-test/parallelize-prog-diags-output

gsutil cp /tmp/$random/diags.nc $OUTPUT/run1/diags.nc
gsutil cp /tmp/$random/metrics.json $OUTPUT/run1/metrics.json

# generate movies for short sample prognostic run
#prognostic_run_diags movie --n_jobs 1 --n_timesteps 2 $RUN /tmp/$random
#stitch_movie_stills.sh /tmp/$random $OUTPUT/run1

# make a second copy of diags/metrics since generate_report.py needs at least two runs
gsutil -m cp -r $OUTPUT/run1 $OUTPUT/run2

# generate report based on diagnostics computed above
prognostic_run_diags report $OUTPUT $OUTPUT

# cleanup
rm -r /tmp/$random

echo "Yay! Prognostic run report integration test passed. You can view the generated report at:"
echo "https://storage.cloud.google.com/vcm-ml-scratch/test-prognostic-report/${random}/index.html"