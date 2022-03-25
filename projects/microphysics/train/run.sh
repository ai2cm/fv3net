#!/bin/bash

set -e
set -o pipefail

group="$(openssl rand -hex 3)"
config_file="gscond.yaml"
model_name="gscond-pressure-in-${group}"

<<<<<<< HEAD
out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
flags="--out_url ${out_url} ${extra_flags}"
argo submit argo.yaml \
    --name "${model_name}" \
    -p training-config="$(base64 --wrap 0 $config_file)" \
    -p flags="$flags" | tee -a submitted-jobs.txt
=======
for config in limited
do
    model_name=cycle-trained-${config}
    config_file=${config}.yaml
    out_url=$(artifacts resolve-url vcm-ml-experiments microphysics-emulation "${model_name}-${group}")
    data_url=gs://vcm-ml-experiments/microphysics-emulation/2022-03-17/online-12hr-cycle-v3-online/artifacts/20160611.000000/netcdf_output
    argo submit argo.yaml \
        --name "${model_name}-${group}" \
        -p training-config="$(base64 --wrap 0 $config_file)"\
        -p flags="--out_url ${out_url} --train_url ${data_url} --test_url ${data_url}" \
        | tee -a experiment-log.txt
done
>>>>>>> origin/master
