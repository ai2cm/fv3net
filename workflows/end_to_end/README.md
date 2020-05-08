## End to End Workflow

Updated March 2020

This workflow serves to orchestrate the current set of VCM-ML workflows into one, 
allowing for going from a completed SHiELD run to a prognostic coarse run with ML
parameterization in a reasonable amount of time and with minimal active oversight.

The follwing steps are currently integrated into the workflow for use in experiments:

- coarsening restarts
- coarsening diagnostics 
- one step runs, i.e., short runs from coarsened restart files
- creating training data
- training an sklearn model
- testing an sklearn model
- a prognostic run of the coarse model using the trained sklearn parameterization
- generation of diagnostics for the prognostic run

The workflow starting point is flexible, i.e., with any of the steps above, as is
its endpoint. If starting at a SHiELD run coarsened to C384, it is required that
the SHiELD C384 restart files and diagnostics are available (locally or remotely).
If starting at a later point, it is assumed that the outputs of all previous steps
are available. 

The workflow syntax, specified in the yaml, is designed to be flexible enough to allow
any set of positional and optional arguments to be sent to a `python argparse` script
interface. Thus, intermediate shell scripts to launch steps are unnecessary and 
discouraged to reduce workflow maintenance. 


### Usage

Call the submit script from the top-level directory of the fv3net repo:


`./workflows/end_to_end/submit_workflow.sh {EXPERIMENT_CONFIG_YAML}`

The config YAML file is the focal point of using the workflow (the submission script
should not need to be modified). An indivudal experiment is run by calling the script
with a particular  configuration. Selection of steps to run, their individual 
configurations, and overall experiment parameters is done in the experiment
configuration YAML.


#### Experiment configuration YAML syntax

Here is an example portion of a YAML file used to configure the workflow and run an experiment:

```
storage_proto: gs
storage_root: vcm-ml-data/orchestration-testing
experiment:
  name: test-experiment
  unique_id: True
  max_stubs: 1
  steps_to_run:
    - coarsen_restarts
    - coarsen_diagnostics
    - one_step_run
    - create_training_data
    - train_sklearn_model
    - test_sklearn_model
    - prognostic_run
    - diags_prognostic_run
  steps_config:
    coarsen_restarts:
      command: python -m fv3net.pipelines.coarsen_restarts
      args:
        data_to_coarsen:
          location: gs://vcm-ml-data/orchestration-testing/shield-C384-restarts-2019-12-04
        grid_spec: 
          location: gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy
        source-resolution: 384
        target-resolution: 48
        --runner: DataflowRunner
    one_step_run:
      command: python workflows/one_step_jobs/orchestrate_submit_jobs.py
      args:
        restart_data:
          from: coarsen_restart
        experiment_yaml: workflows/one_step_jobs/all-physics-off.yml
        docker_image: us.gcr.io/vcm-ml/prognostic-run-orchestration
        --n-steps: 50
    ...
```

##### Top-level arguments:

- **storage_proto**: protocol of the filesystem; can be either 'gs' or 'file'
- **storage_root**: root directory of storage for the experiment

##### Experiment level arguments:

- **name**: name of experiment
- **unique_id**: a UUID is appended to experiment name if True; if false, the `name` field is used directly. True is reccommended for safety as it will not be likely to clobber old experiments, whereas False allows for specifying an existing experiment to rerun or start from an intermediate step.
- **max_stubs**: (Optional) If supplied, the number of args to include in the output step directory names. If not supplied, defaults to 0 (step name only). 
- **steps_to_run**: a YAML list of workflow steps to execute for this experiment

##### Step level arguments:

For each step in `steps_to_run`, its configuration is set by its equivalently named block in `step_config`. Step configuration can still be defined here but it will not be executed if it is excluded from `steps_to_run`. *Note:* the order of parameters correspond to the positional commandline arguments appended to the job `command`.  Parameters are translated to arguments in the order of _args_, _output_. 

- **command**: command to execute the step, e.g., a python or bash command and script that executes the step (without its arguments)
- **args**: a dict of positional and optional argument key-value pairs for the _command_ above. Arguments are used to specify step input data, configuration files, and parameters. Values may be literals, names of previous steps which to use output as input for the current step, paths to pre-existing data sources, or lists of multiple values for a given key; each value in the args dict must be one of the following types:
  - **value/hashable**: A single argument value
  - **dict**: if the arg value is a dict, it is for specifying the source of input data to the step. Supported keys in the dict are as follows (only one may be used for a given step argument):
    - **from**: name of the previous step which produces the input data for this step, e.g., `coarsen_restarts` is an input to `one_step_run`. Whatever the output location is from that step will be used. The previous step must be run as part of the experiment. 
    - **location**: explicit path to required input data for this step, which has been generated prior to the current experiment.
  - **list**: multiple values to be passed to the command under the same argument name. This is useful for the `--extra-packages` argument in dataflow jobs which may need to be called on multiple packages. 
  Argument keys may begin with `--` to denote an optional argument. Optional arguments will be appended to the end of the command as --{param_keyname} {param_value}. Arguments without the leading `--` are treated as positional arguments.
  
  - **Argo workflow parameters**: Arguments for argo submit steps can be passed with the key given as `"-p <parameter_name>="`. See the Argo workflow steps section below for more info.
  
- **output_location**: (Optional) explicit path to store output from this step.  This parameter is autogenerated if not provided and is used as the source for any input `from` refs.  
    

#### Data output locations

If no `output_location` is specified for a step,  it will be output via the following structure:

```{storage_proto}://{storage_root}/{experiment_name}/{step_name}```

where `experiment_name` is the name plus the UUID (if added), and the `step_name` is defined as the name of the workflow step with the first 3 extra_args key/values appended to it.


#### The dataflow `--runner` argument

For steps using Apache Beam and Google Dataflow, the `--runner` optional argument can be passed. This is a reserved argument key that accepts only two values: `DirectRunner` and `DataflowRunner`. If `DirectRunner`is used, this is passed as the `--runner` value to launch a local job. If `DataflowRunner` is passed, a set of Dataflow arguments are appended to the job submission script to enable launching Dataflow jobs on GCS. Those arguments are stored for specific steps which require them in `workflows/end_to_end/dataflow.py`. 


### Creating new workflow steps

To write a new step in the workflow, create a CLI for the step that follows the following format:

```{COMMAND} {POSITIONAL_ARG1} {POSITIONAL_ARG2} ... {POSITIONAL_ARGN} {OUTPUT_PATH} {--OPTIONAL_ARG} {OPTIONAL_ARG_VALUE} ...```

then add the step to the config YAML file, in both the `steps_to_run` list and the `steps_config` dict. At a minimum, the `command` and `args` values must be specified for the step configuration. Additionally, the step must be listed in `steps_to_run` in the order in which it is necessary, i.e., after any steps upon which it depends for input, and before any steps that depend on it for output. 

### Preparing Kubernetes jobs with Kustomize

The directory `kustomization` contains a base set of kubernetes manifests and
fv3net configurations that can be used to run reproducible end to end
experiments with kubectl alone. 

[kustomize] is a powerful templating system that is packaged with kubectl. It
works by specifying a base set of resources,
and then allows other workflows to inherit and modify this base configuration
in a variety of ways (e.g. add configurations to a configmap, or add a suffix
to the k8s job). Configurations to individual workflow steps are mostly
controlled by the `.yml` files referred to within the `base/end_to_end.yaml`
file. The following resources must be configured by editing the
`kustomization.yaml` file in the root of the template directory. The settings
in this file are overlayed on top of the configurations in
`base/kustomization.yaml`. So go to that file to change settings shared by
all the experiments (e.g. the prognostic run image tag and fv3net image tag).

See [this repo](https://github.com/VulcanClimateModeling/vcm-workflow-control) for an
example of how to use this base configuration.


[kustomize]: https://kustomize.io/ 


### Argo workfow steps
Argo workflows can be run as steps in the end to end pipeline; the command value is the `argo submit` statement and parameters for Argo can be specified in the step's entry in the end to end yalm in a very similar fashion as for python commands. The only difference is that Argo parameter names should be given as `"-p <parameter_name>="`. For example,
```
    test_sklearn_model: 
      command: argo submit test_sklearn_model.yaml
      args:
        "-p trained_model=":
          from: train_sklearn_model
        "-p testing_data=":
          from: create_training_data
        "-p diagnostics_data=":
          location: $C48_DIAGNOSTICS
        "-p variable_filename=": $CONFIG/test_sklearn_variable_names.yml 
```
will run as the following full command: `argo submit test_sklearn_model.yaml -p trained_model=<path from train_sklearn_model_output> ...`.