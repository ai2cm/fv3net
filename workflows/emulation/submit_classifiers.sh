#!/bin/bash

argo_workflow=$1
subs_file=$2

var_arr=(
    dv_output
    du_output
    tdt_update 
    # rtg_output_0
    # rtg_output_1
    # rtg_output_2
    # rtg_output_3
    # rtg_output_4
    # rtg_output_5
    # rtg_output_6
    # rtg_output_7
)

workdir=$(mktemp -d)

for var in "${var_arr[@]}"
do
    export VARNAME=$var
    # export DATESTR=$(date "+%Y-%m-%d")
    export DATESTR=2020-12-30
    envsubst < $subs_file > $workdir/params.yaml
    # argo submit /home/andrep/repos/fv3net/workflows/argo/emulation.yaml -f $workdir/params.yaml
    argo submit $argo_workflow -f $workdir/params.yaml
done

