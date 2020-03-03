DATA_PATH=$1
DIAG_PATH=$2
OUTPUT_PATH=$3
TIMESTEPS_PER_OUTPUT=$4
TRAIN_FRACTION=$5

user=$(whoami)
user=${user,,}

python -m fv3net.pipelines.create_training_data \
${DATA_PATH} \
${DIAG_PATH} \
${OUTPUT_PATH} \
--timesteps-per-output-file ${TIMESTEPS_PER_OUTPUT} \
--train-fraction ${TRAIN_FRACTION} \
--job_name create-training-data-${user} \
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
