#!/bin/bash

python -m fv3config.fv3run._native_main \
"[[\"$CONFIG\", \"$RUNDIR\"], {\"runfile\": \"$RUNFILE\", \"capture_output\": false}]" \
|& tee -a  "$RUNDIR/logs.txt"
