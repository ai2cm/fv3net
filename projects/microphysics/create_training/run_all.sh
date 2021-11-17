#!/bin/bash

set -e

PROG_YAML="./fv3config.yaml"
PROJECT="microphysics-emu-data"

./run_single.sh ${PROG_YAML} 01 ${PROJECT}
./run_single.sh ${PROG_YAML} 02 ${PROJECT}
./run_single.sh ${PROG_YAML} 03 ${PROJECT}
./run_single.sh ${PROG_YAML} 04 ${PROJECT}
./run_single.sh ${PROG_YAML} 05 ${PROJECT}
./run_single.sh ${PROG_YAML} 06 ${PROJECT}
./run_single.sh ${PROG_YAML} 07 ${PROJECT}
./run_single.sh ${PROG_YAML} 08 ${PROJECT}
./run_single.sh ${PROG_YAML} 09 ${PROJECT}
./run_single.sh ${PROG_YAML} 10 ${PROJECT}
./run_single.sh ${PROG_YAML} 11 ${PROJECT}
./run_single.sh ${PROG_YAML} 12 ${PROJECT}
