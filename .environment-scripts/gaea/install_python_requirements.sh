#!/bin/bash

ENVIRONMENT_SCRIPTS_DIR=$1
REQUIREMENTS_PATH=$2

# Setting CC=cc and MPICC=cc ensures that the native compiler is used
# when installing mpi4py
CC=cc MPICC=cc bash $ENVIRONMENT_SCRIPTS_DIR/install_python_requirements.sh $REQUIREMENTS_PATH
