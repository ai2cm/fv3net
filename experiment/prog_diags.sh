#!/bin/bash

set -e
argo submit --from=workflowtemplates/prognostic-run-diags \
    -p runs="$(< configs/rundirs.json)" \
    -p flags="--verification nudged_c48_fv3gfs_2016" \
    -p make-movies=false \
    --name n2obs-nn-random-seed
