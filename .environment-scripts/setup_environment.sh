#!/bin/bash
set -e

# Note if this script is modified the base image will need to be rebuilt.

INSTALL_TYPE=$1  # Can be one of "all", "base", "fv3gfs-fortran", "python-requirements", "wrapper", or "fv3net-packages"
PLATFORM=$2
CLONE_PREFIX=$3
INSTALL_PREFIX=$4
FV3NET_DIR=$5
CALLPYFORT=$6  # Should be "" if not installed
CONDA_ENV=$7  # Optional (not needed in prognostic run docker image)


# Note modules, once loaded, are global in scope, so we load them here.
SCRIPTS=$FV3NET_DIR/.environment-scripts
MODULES_FILE=$FV3NET_DIR/external/fv3gfs-fortran/FV3/conf/modules.fv3.$PLATFORM
if [ -f "$MODULES_FILE" ];
then
    source "$MODULES_FILE"
fi

mkdir -p "$CLONE_PREFIX"
mkdir -p "$INSTALL_PREFIX"

if [ "$INSTALL_TYPE" == "all" ] || [ "$INSTALL_TYPE" == "base" ];
then
    bash "$SCRIPTS"/setup_base_environment.sh \
        "$PLATFORM" \
        "$CLONE_PREFIX" \
        "$INSTALL_PREFIX" \
        "$FV3NET_DIR" \
        "$CALLPYFORT" \
        "$CONDA_ENV"
fi

if [ "$INSTALL_TYPE" != "base" ];
then
    bash "$SCRIPTS"/setup_development_environment.sh \
        "$INSTALL_TYPE" \
        "$PLATFORM" \
        "$CLONE_PREFIX" \
        "$INSTALL_PREFIX" \
        "$FV3NET_DIR" \
        "$CALLPYFORT" \
        "$CONDA_ENV"
fi
