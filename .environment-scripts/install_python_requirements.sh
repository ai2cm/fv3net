#!/bin/bash
set -e

REQUIREMENTS_PATH=$1

pip install --no-cache-dir -r $REQUIREMENTS_PATH
