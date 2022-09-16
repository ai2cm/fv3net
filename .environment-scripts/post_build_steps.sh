#!/bin/bash
set -e

FV3NET_DIR=$1

# Make microphysics scripts executable
chmod +x $FV3NET_DIR/projects/microphysics/scripts/*
