INPUT_DATA_PATH=$1
OUTPUT_DATA_PATH=$2
TRAINING_CONFIG_PATH=$3
MASK_TO_SURFACE_TYPE=$4

python -m fv3net.regression.sklearn.train \
  ${INPUT_DATA_PATH}"/train" \
  ${TRAINING_CONFIG_PATH} \
  ${OUTPUT_DATA_PATH} \
  --mask-to-surface-type ${MASK_SURFACE_TYPE}
