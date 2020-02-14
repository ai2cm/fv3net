MODEL_PATH=$1
DATA_PATH=$2
DIAGS_PATH=$3
OUTPUT_PATH=$4
NUM_TEST_ZARRS=$5


python -m fv3net.regression.model_diagnostics \
  --model-path ${MODEL_PATH} \
  --test-data-path ${DATA_PATH}"/test" \
  --high-res-data-path ${DIAGS_PATH} \
  --num-test-zarrs ${NUM_TEST_ZARRS}
