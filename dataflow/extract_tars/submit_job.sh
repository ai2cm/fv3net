# !/bin/sh

# GCS prefixes under gs://vcm-ml-data/ to retrieve tars and place output
SOURCE_TAR_PREFIX='2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data'
OUTPUT_DESTINATION='2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted'

# Worker configuration
NUM_WORKERS='10'
MAX_NUM_WORKERS='40'

python -m extractflow \
    $SOURCE_TAR_PREFIX \
    $OUTPUT_DESTINATION \
    --job_name test-work-$(whoami) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --max_num_workers $MAX_NUM_WORKERS \
    --disk_size_gb 80 \
    --type_check_strictness 'ALL_REQUIRED' \
    --worker_machine_type n1-standard-1

