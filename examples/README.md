# Example workflow configuration and submission scripts

This directory contains a number of example workflow configurations and
related submission scripts. 

## Submission

See the `Makefile` in this directory for the available workflow submission rules.

## Image tags and workflow versions

To use a particular tag of any image required by the workflows, modify the
`kustomization.yaml` found in this directory. To use the latest argo workflows,
update the `examples/fv3net` submodule.
