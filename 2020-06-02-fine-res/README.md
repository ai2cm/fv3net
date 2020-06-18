Fine-resolution Q1/Q2 Workflow
==============================

This directory contains kubernetes manifests to run each step of the end to
end fine-res budget workflow. This workflow was developed incrementally, so
it does not use any high-level orchestration. However, each operation is
wrapped in a K8s manifest to ensure full reproducibility.

Reproducing
-----------

To start, deploy all the shared configurations (output GCS URLs) to the
cluster:

    kubectl apply -f output-configmap.yaml

Once that is complete, run each of the following steps in sequence, waiting
for the previous one to complete:

    kubectl apply -f 0_restarts_to_zarr.yaml
    kubectl apply -f 1_fine_res_job.yaml


Saving the coarsen-restarts
---------------------------

While the fine res Q1 and Q2 do not depend on the coarsened restart data,
this folder contains a yaml file computing them on the new ShiELD data. This
job can be executed using

    kubectl apply -f coarsen_restarts.yaml


Where are the outputs?
----------------------

See [output-configmap.yaml] for an up-to-date list of file locations.

