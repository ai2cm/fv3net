# VCM Experiment Configurations

## Purposes

This repository contains *declarative* configurations of our workflow, which should be cleanly separated from the source code in [fv3net]. Decoupling our source from the configuration will allow us to run experiments against different versions of the fv3net source. As the workflows in that repo become more decoupled and plug-n-play, this will allow us to change swap components out easily without being bound to the current master version of [fv3net].

## Allowed Dependencies

We should be able to manage our workflow with the following tools only:
- bash
- jq
- envsubst
- kubectl

In particular, managing dependencies with python is very difficult and leads to giant docker images, so python should not be required to *submit* any workflows. Kubernetes provides a very rich declarative framework for managing computational resources, so we should not need any other tools. In the future, we may see if including python and a very minimal dependency set (e.g. kubernetes API, yaml, jinja2) will be helpful. 

[fv3net]: https://github.com/VulcanClimateModeling/fv3net