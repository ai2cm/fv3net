INPUT_DATA_PATH=$1
OUTPUT_DATA_PATH=$2
TRAINING_CONFIG_PATH=$3

python -m fv3net.regression.sklearn.train \
  --train-data-path ${INPUT_DATA_PATH} \
  --train-config-file ${TRAINING_CONFIG_PATH} \
  --output-data-path ${OUTPUT_DATA_PATH}
