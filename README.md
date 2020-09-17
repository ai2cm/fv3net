# VCM Experiment Configurations

## Purposes

This repository contains *declarative* configurations of our workflow, which
should be cleanly separated from the source code in [fv3net]. Decoupling our
source from the configuration will allow us to run experiments against
different versions of the fv3net source. As the workflows in that repo become
more decoupled and plug-n-play, this will allow us to change swap components
out easily without being bound to the current master version of [fv3net].

## Dependencies

To ensure that the configurations in this repository are easy to review and
reproducible only a limited set of dependencies are allowed. In particular,
python is not an allowed dependency, since that could lead to complicated
configuration transformation scripts. This repo should only require these
tools:

- bash (4.4.2)
- kubectl (1.18.2)
- jq (1.5.1)
- yq (https://github.com/kislyuk/yq) (2.10.0)

## Workflow

`examples/` contains clean examples of common workflows, that you can easily
base your analysis from. Currently, only the [complete
training-diagnosics-prognostic-run](examples/train-evaluate-prognostic-run)
workflow has a good example.



[fv3net]: https://github.com/VulcanClimateModeling/fv3net
