#!/bin/bash

SETUP_PY=../../setup.py
VCM_PATH=../../external/vcm
# Have a few more workers than the 96 input files present to avoid having all
# the workers wait for a single file to download
NUM_WORKERS=100
BEAM_MAIN_MODULE=fv3net.pipelines.coarsen_surface_c3072

# build the sdist for vcm
#
# Notes:
# Calling the setup.py from this current directory and then using the tarball from the
# local dists folder led to import errors when submitting to the remote
# Therefore, we run the setup.py for vcm in the external/vcm directory.
(
cd $VCM_PATH
python setup.py sdist
)


# submit the job
python -m $BEAM_MAIN_MODULE  \
    --job_name test-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --setup_file $SETUP_PY \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --max_num_workers $NUM_WORKERS \
    --disk_size_gb 100 \
    --worker_machine_type n1-standard-2 \
    --extra_package $VCM_PATH/dist/vcm*.tar.gz
