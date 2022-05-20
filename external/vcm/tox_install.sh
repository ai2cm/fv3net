#!/usr/bin/env bash

set -e -x

pip install -c ../../constraints.txt numpy
pip install -c ../../constraints.txt $@
pip install .
