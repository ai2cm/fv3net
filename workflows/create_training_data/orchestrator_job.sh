INPUT_LOCATION=$1
OUTPUT_LOCATION=$2
DIAG_PATH=$3
TIMESTEPS_PER_OUTPUT=$4
MASK_SURFACE_TYPE=$5
TRAIN_FRACTION=$6

python -m fv3net.pipelines.create_training_data \
--gcs-input-data-path ${INPUT_LOCATION} \
--diag-c48-path ${DIAG_PATH} \
--timesteps-per-output-file ${TIMESTEPS_PER_OUTPUT} \
--mask-to-surface-type ${MASK_SURFACE_TYPE} \
--train-fraction ${TRAIN_FRACTION} \
--gcs-output-data-dir ${OUTPUT_LOCATION} \
--gcs-bucket gs://vcm-ml-data \
--job_name create-training-data-$(whoami) \
--project vcm-ml \
--region us-central1 \
--runner DataflowRunner \
--temp_location gs://vcm-ml-data/tmp_dataflow \
--num_workers 4 \
--max_num_workers 30 \
--disk_size_gb 30 \
--worker_machine_type n1-standard-1 \
--setup_file ./setup.py \
--extra_package external/vcm/dist/vcm-0.1.0.tar.gz
