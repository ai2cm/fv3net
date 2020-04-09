# !/bin/sh

# GCS prefixes under gs://vcm-ml-data/ to retrieve tars and place output
# SOURCE_TAR_PREFIX='2019-10-05-X-SHiELD-C3072-to-C384-re-uploaded-restart-data'
SOURCE_TAR_PREFIX='test_dataflow'
OUTPUT_DESTINATION='2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted'

# Worker configuration
NUM_WORKERS='4'
MAX_NUM_WORKERS='40'

user=$(whoami)
user=${user,,}

python -m fv3net.pipelines.extract_tars \
    $SOURCE_TAR_PREFIX \
    $OUTPUT_DESTINATION \
    --job_name test-extract-${user} \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers $NUM_WORKERS \
    --max_num_workers $MAX_NUM_WORKERS \
    --disk_size_gb 80 \
    --type_check_strictness 'ALL_REQUIRED' \
    --worker_machine_type n1-standard-1 \
    --setup_file ./setup.py \
    --extra_package external/vcm/dist/vcm-0.1.1.tar.gz

