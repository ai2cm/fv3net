#!/bin/bash
set -e

INSTALL_DIR=$INSTALL_PREFIX/software/bats_dest/
CLONE_DIR=$CLONE_PREFIX/software/bats_clone/

URL=https://github.com/bats-core/bats-core.git

git clone $URL $CLONE_DIR

bash $CLONE_DIR/install.sh $INSTALL_DIR


