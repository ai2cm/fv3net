#!/bin/bash

PROGNOSTIC_RUN_IMAGE=$(make -s image_name_prognostic_run VERSION=$1)
FV3NET_IMAGE=$(make -s image_name_fv3net VERSION=$1)
JOB_YML="submit_e2e_job_k8s.yml"
E2E_CONFIG_YML="end_to_end_configuration.yml"

# create yaml with unique testing job name
cd tests/end_to_end_integration
rand_tag=$(openssl rand -hex 6)
job_name=$(cat ./.submit_template/$JOB_YML | yq r - metadata.name)
new_job_name=${job_name}-${rand_tag}

# save with new job name and correct image tag
yq w ./.submit_template/$JOB_YML metadata.name $new_job_name > $JOB_YML
yq w -i $JOB_YML spec.template.spec.containers[0].image $FV3NET_IMAGE
yq w ./.submit_template/$E2E_CONFIG_YML \
    experiment.steps_config.one_step_run.args.docker_image $PROGNOSTIC_RUN_IMAGE \
    > $E2E_CONFIG_YML
yq w -i $E2E_CONFIG_YML \
    experiment.steps_config.prognostic_run.args.docker_image $PROGNOSTIC_RUN_IMAGE