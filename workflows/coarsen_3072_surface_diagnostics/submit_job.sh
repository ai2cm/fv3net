#!/bin/bash

VCM_PATH=../../external/vcm
NUM_WORKERS=1
BEAM_MAIN_MODULE=fv3net.pipelines.coarsen_surface_c3072

# build the sdist for vcm
python $VCM_PATH/setup.py sdist


# submit the job
python -m $BEAM_MAIN_MODULE  \
    --job_name test-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --max_num_workers $NUM_WORKERS \
    --disk_size_gb 100 \
    --worker_machine_type n1-standard-2 \
    --extra_package dist/vcm*.tar.gz
