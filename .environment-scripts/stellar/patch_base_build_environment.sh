#!/bin/bash

CLONE_PREFIX=$1

# 'bats' executable destination. Necessary to build FMS library
export PATH=$CLONE_PREFIX/bats-core/bin:$PATH
