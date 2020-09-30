#!/bin/bash

set -xe

RUN=gs://vcm-ml-code-testing-data/sample-prognostic-run-output

random=$(openssl rand --hex 6)
OUTPUT=gs://vcm-ml-scratch/test-prognostic-report/$random

cd workflows/prognostic_run_diags

# compute diagnostics/mterics for a short sample prognostic run
mkdir -p /tmp/$random
python save_prognostic_run_diags.py $RUN /tmp/$random/diags.nc
python metrics.py /tmp/$random/diags.nc > /tmp/$random/metrics.json
gsutil cp /tmp/$random/diags.nc $OUTPUT/run1/diags.nc
gsutil cp /tmp/$random/metrics.json $OUTPUT/run1/metrics.json

# generate movies for short sample prognostic run
python generate_movie_stills.py --n_jobs 1 --n_timesteps 2 $RUN /tmp/$random
bash stitch_movie_stills.sh /tmp/$random $OUTPUT/run1

# make a second copy of diags/metrics since generate_report.py needs at least two runs
gsutil -m cp -r $OUTPUT/run1 $OUTPUT/run2

# generate report based on diagnostics computed above
python generate_report.py $OUTPUT $OUTPUT/index.html

# cleanup
rm -r /tmp/$random