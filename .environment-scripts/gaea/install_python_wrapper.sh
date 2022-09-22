#!/bin/bash

ENVIRONMENT_SCRIPTS_DIR=$1
FV3_DIR=$2

CC=cc LDSHARED='cc -shared' bash $ENVIRONMENT_SCRIPTS_DIR/install_python_wrapper.sh $FV3_DIR
