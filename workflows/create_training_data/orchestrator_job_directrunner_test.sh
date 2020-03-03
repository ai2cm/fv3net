DATA_PATH=$1
DIAG_PATH=$2
OUTPUT_PATH=$3
TIMESTEPS_PER_OUTPUT=$4
MASK_SURFACE_TYPE=$5
TRAIN_FRACTION=$6

python -m fv3net.pipelines.create_training_data \
${DATA_PATH} \
${DIAG_PATH} \
${OUTPUT_PATH} \
--timesteps-per-output-file ${TIMESTEPS_PER_OUTPUT} \
--mask-to-surface-type ${MASK_SURFACE_TYPE} \
--train-fraction ${TRAIN_FRACTION} \
--runner DirectRunner