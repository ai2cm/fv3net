#!/bin/bash
set -e

INSTALL_DIR=$HOME/software/bats_dest/
CLONE_PREFIX=$HOME/software/bats_clone/

URL=https://github.com/bats-core/bats-core.git

git clone $URL $CLONE_PREFIX

bash $CLONE_PREFIX/install.sh $INSTALL_DIR


