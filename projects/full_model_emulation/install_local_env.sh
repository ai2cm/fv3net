#/bin/bash

set -e -x

ENV_NAME=torch

# PROJECT_NAME=${ENV_NAME} make -C ../../ create_environment
source activate ${ENV_NAME}
# conda install mkl=2022.0.1 $ optional for intel-based computers
conda install pytorch=1.12.0 cudatoolkit=11.3 -c pytorch
