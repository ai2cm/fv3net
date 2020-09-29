#!/bin/bash

set -xe

#RUN=gs://vcm-ml-code-testing-data/sample-prognostic-run-output
RUN=gs://vcm-ml-scratch/oliwm/prog-testing-output
random=$(openssl rand --hex 6)
OUTPUT=gs://vcm-ml-scratch/test-prognostic-report/$random

cd workflows/prognostic_run_diags

# compute diagnostics for a short sample prognostic run
bash entrypoint.sh $RUN $OUTPUT/run1

# generate movies for short sample prognostic run
bash entrypoint_movie.sh $RUN $OUTPUT/run1 "--n_jobs 1 --n_timesteps 2"

# make a second copy of diags, since generate_report.py needs at least two runs
gsutil -m cp -r $OUTPUT/run1 $OUTPUT/run2

# generate report based on diagnostics computed above
python generate_report.py $OUTPUT $OUTPUT/index.html
