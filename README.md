# VCM Experiment Configurations

## Purposes

This repository contains *declarative* configurations of our workflow, which should be cleanly separated from the source code in [fv3net]. Decoupling our source from the configuration will allow us to run experiments against different versions of the fv3net source. As the workflows in that repo become more decoupled and plug-n-play, this will allow us to change swap components out easily without being bound to the current master version of [fv3net].

## Allowed Dependencies

We should be able to manage our workflow with the following tools only:

- bash: GNU bash, version 4.4.20(1)-release (x86_64-pc-linux-gnu)
- jq: jq-1.5-1-a5b5cbe
- envsubst: envsubst (GNU gettext-runtime) 0.19.8.1.
    installed via `apt-get install gettext`
- kubectl - v1.16.3


The idea is that these are stable and robust tools that are easy to install. OTOH, managing dependencies with python is very difficult and leads to giant docker images, so python should not be required to *submit* any workflows. Kubernetes provides a very rich declarative framework for managing computational resources, so we should not need any other tools. In the future, we may see if including python and a very minimal dependency set (e.g. kubernetes API, yaml, jinja2) will be helpful. 

## Structure

``` 
.
├── CODEOWNERS
├── Makefile
├── README.md
├── end_to_end
│   ├── configs
│   │   └── integration-test
│   │       ├── coarsen_c384_diagnostics_integration.yml
│   │       ├── create_training_data_variable_names.yml
│   │       ├── diag_table_prognostic
│   │       ├── one_step_jobs_integration.yml
│   │       ├── prognostic_run_integration.yml
│   │       ├── test_sklearn_variable_names.yml
│   │       └── train_sklearn_model.yml
│   ├── end_to_end.yml
│   ├── generate.sh
│   ├── generate_all_configs.sh
│   ├── job_template.yml
│   └── run_integration_with_wait.sh
├── jobs
└── manifests
    └── integration-test-service-account.yml

```

- experiment-name cannot be an underscore since this name is used to name the k8s job, and k8s does not allow underscore in its metadata.names attributes.
- manifests: these are k8s configurations that change infrequently such as permissions
- generated jobs go into jobs. This folder is deleted with every invocation of `make generate_configs`.
    - It should be checked into version control for complete reproducibility

## Workflows

### Production

1. Create a new configuration under end_to_end/configs
1. Run `make generate_configs` to save these manifests for all the configs and save them into the jobs folder.
1. Check in the manifests folder to version control, and push these changes to master

### Development

1. Create a new configuration under end_to_end/configs
1. run script `end_to_end/generate.sh` from the to view the generated manifests, pipe this output to disk if you want to use it
1. Submit scripts using `kubectl apply`

## Troubleshooting

This workflow *might* work if your versions deviate from the ones listed above. If you run into issues, run

    make versions

to show how your versions deviate from these.

[fv3net]: https://github.com/VulcanClimateModeling/fv3net
