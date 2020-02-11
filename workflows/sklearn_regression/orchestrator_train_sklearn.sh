TRAIN_DATA_PATH=$1
OUTPUT_PREFIX=$2
TRAINING_CONFIG_PATH=$3
REMOTE_OUTPUT_URL=$4

python -m fv3net.regression.sklearn.train \
  --train-config-file ${TRAINING_CONFIG_PATH} \
  --output-dir-suffix ${OUTPUT_PREFIX} \
  --train-data-path ${TRAIN_DATA_PATH} \
  --remote-output-url ${REMOTE_OUTPUT_URL}
