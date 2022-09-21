#!/bin/bash

ENVIRONMENT_SCRIPTS_DIR=$1
CLONE_PREFIX=$2
INSTALL_PREFIX=$3

# CC=cc is required to specify the C++ compiler use
CC=cc bash $ENVIRONMENT_SCRIPTS_DIR/install_esmf.sh $CLONE_PREFIX $INSTALL_PREFIX Unicos intel default
