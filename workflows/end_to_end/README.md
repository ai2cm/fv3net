## End to End Workflow

February 2020

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

The workflow starting point is flexible, i.e., with any of the steps above, as is
its endpoint. If starting at a SHiELD run coarsened to C384, it is required that
the SHiELD C384 restart files and diagnostics are available (locally or remotely).
If starting at a later point, it is assumed that the outputs of all previous steps
are available. 


### Usage

Call the submit script from the top-level directory of the fv3net repo:


`./workflows/end_to_end/submit_workflow.sh {EXPERIMENT_CONFIG_YAML}`

The config YAML file is the focal point of using the workflow (the submission script
should not need to be modified). An indivudal experiment is run by calling the script
with a particular  configuration. Selection of steps to run, their individual 
configurations, and overall experiment parameters is done in the experiment
configuration YAML.


### Experiment configuration YAML syntax

Here is a portion of the YAML file to configure the workflow to run an experiment:

```
user: brianh
storage_proto: 'gs'
storage_root: vcm-ml-data/orchestration-testing
experiment:
  name: test-experiment
  unique_id: True
  fv3net_hash: ca279cb79c0b97a5e5e758ec8785612b3c7ec044
  steps_to_run:
    - coarsen_restarts
    - coarsen_diagnostics
    - one_step_run
    - create_training_data
    - train_sklearn_model
    - test_sklearn_model
    - prognostic_run
  experiment_vars:
    one_step_yaml: workflows/one_step_jobs/all-physics-off.yml
  steps:
    coarsen_restarts:
      command: workflows/coarsen_restarts/orchestrator_job.sh
      inputs:
        data_to_coarsen:
          location: gs://vcm-ml-data/orchestration-testing/shield-C384-restarts-2019-12-04
      method:
        source-resolution: 384
        target-resolution: 48
    one_step_run:
      command: python ./workflows/one_step_jobs/orchestrate_submit_jobs.py
      inputs:
        restart_data:
          from: coarsen_restarts
          location: 
      method: 
        experiment_yaml: one_step_yaml
        experiment_label: test-orchestration-group
      config_transforms:
        add_unique_id: ['experiment_label']
        use_top_level: ['experiment_yaml']
```

#### Top-level arguments:

- user
- storage_proto: protocol of the filesystem; can be either 'gs' or 'file'
- storage_root: root directory of storage for the experiment

#### Experiment level arguments:

- name: name of experiment
- unique_id: a UUID is appended to experiment name if True; if false, the `name`field is used directly. True is used for new workflows, whereas False allows for specifying an existing experiment to rerun or start from an intermediate step.
- fv3net_hash: Repo commit hash of the fv3net codebase to be used (not yet implemted)
- steps_to_run: a YAML list of workflow steps to execute for this experiment
- experiment_vars: a method that applies to more than one step in the experiment can be set here; see `use_top_level` under step configuration

#### Step level arguments:

For each step in the workflow, configuration is set by its block in `step_config`, with the name of the block being the name of the step corresponding to its name in `steps_to_run`. Step configuration may be defined here but not executed if it is excluded from `steps_to_run`.

- command: command to execute the step, e.g., a python or bash command and script that executes the step (without its arguments)
- inputs: a list of input data types required by the step; for each input data type, _one_ (but not both) of the following must be specified:
    - from: name of the previous step which produces the input data for this step, e.g., `coarsen_restarts` is an input to `one_step_run`. The input step must be run in the same experiment as the current step if `from` is used.
    - location: path to required input data for this step, which has been generated prior to the current experiment
- method: configurable key-value pairs specific to the workflow step; depending on the step, these may be actual method parameters, or the method value may be a step-specific YAML file with further configuration details
- config_transforms: additional functionality useful for Kubernetes FV3GFS model runs:
    - add_unique_id: a list of step method keys for which the experiment-level UUID should be appended to the method value; useful for appending the experiment UUID to the `experiment label` tag appended to Kubernetes jobs
    - use_top_level: a list of step method keys for which the method value should be remapped using `experiment_vars` defined above; this allows for setting multiple steps' methods to the same value, e.g., using the same one-step YAML config file for both the one-step and prognostic jobs.
    

### Data output locations

Data from each step are output via the following structure:

```{storage_proto}://{storage_root}/{experiment_name}/{step_name}```

where `experiment_name` is the name plus the UUID (if added), and the `step_name` is defined as the name of the workflow step with the first 3 method values appended to it. 