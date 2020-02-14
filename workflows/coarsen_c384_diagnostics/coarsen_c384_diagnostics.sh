INPUT_PATH=$1
OUTPUT_PATH=$2
CONFIG_PATH=$3

python workflows/coarsen_c384_diagnostics/coarsen_c384_diagnostics.py \
--input-path=$INPUT_PATH \
--output-path=$OUTPUT_PATH \
--config-path=$CONFIG_PATH

