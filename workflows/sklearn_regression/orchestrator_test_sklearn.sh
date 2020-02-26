MODEL_PATH=$1
DATA_PATH=$2
DIAGS_PATH=$3
OUTPUT_PATH=$4
NUM_TEST_ZARRS=$5


python -m fv3net.diagnostics.sklearn_model_performance \
  ${MODEL_PATH} \
  ${DATA_PATH}"/test" \
  ${DIAGS_PATH} \
  ${OUTPUT_PATH} \
  --num-test-zarrs ${NUM_TEST_ZARRS}
