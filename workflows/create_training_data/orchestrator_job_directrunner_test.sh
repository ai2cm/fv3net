DATA_PATH=$1
DIAG_PATH=$2
OUTPUT_PATH=$3
TIMESTEPS_PER_OUTPUT=$4
MASK_SURFACE_TYPE=$5
TRAIN_FRACTION=$6

python -m fv3net.pipelines.create_training_data \
--gcs-input-data-path ${DATA_PATH} \
--diag-c48-path ${DIAG_PATH} \
--timesteps-per-output-file ${TIMESTEPS_PER_OUTPUT} \
--mask-to-surface-type ${MASK_SURFACE_TYPE} \
--train-fraction ${TRAIN_FRACTION} \
--gcs-output-data-dir ${OUTPUT_PATH} \
--gcs-bucket gs://vcm-ml-data \
--runner DirectRunner