INPUT_DATA_PATH=$1
OUTPUT_DATA_PATH=$2
TRAINING_CONFIG_PATH=$3

python -m fv3net.regression.sklearn.train \
  ${INPUT_DATA_PATH}"/train" \
  ${TRAINING_CONFIG_PATH} \
  ${OUTPUT_DATA_PATH}
