INPUT_LOCATION=$1
OUTPUT_LOCATION=$2
MODEL_RESOLUTION=$3
DIAG_PATH=$4
TIMESTEPS_PER_OUTPUT=$5
MASK_SURFACE_TYPE=$6
TRAIN_FRACTION=$7

python -m fv3net.pipelines.create_training_data \
--gcs-input-data-path ${INPUT_LOCATION}/one_step_output/C${MODEL_RESOLUTION} \
--diag-c48-path ${DIAG_PATH} \
--timesteps-per-output-file ${TIMESTEPS_PER_OUTPUT} \
--mask-to-surface-type ${MASK_SURFACE_TYPE} \
--train-fraction ${TRAIN_FRACTION} \
--gcs-output-data-dir ${OUTPUT_LOCATION} \
--gcs-bucket gs://vcm-ml-data \
--runner DirectRunner