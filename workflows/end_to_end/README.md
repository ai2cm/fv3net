## End to End Workflow

February 2020

This workflow serves to orchestrate the current set of VCM-ML workflows into one, 
allowing for going from a completed SHiELD run to a prognostic coarse run with ML
parameterization in a reasonable amount of time and with minimal active oversight.

The follwing steps are currently integrated into the workflow:

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


### Basic usage

Called from the top-level directory of the fv3net repo:


`./workflows/end_to_end/submit_workflow.sh ./workflows/end_to_end/{WORKFLOW_CONFIG_YAML}`

The config YAML file is the focal point of using the workflow (the submission script
should not need to be modified). Selection of steps to run, their individual 
configurations, and overall experiment parameters is done in the YAML.


### Configuration YAML syntax

The YAML has the following structure:

```
user: brianh
storage_proto: 'gs'
storage_root: vcm-ml-data/orchestration-testing
experiment:
  name: test-experiment-d8d2f94f
  unique_id: False
  fv3net_hash: ca279cb79c0b97a5e5e758ec8785612b3c7ec044
  # workflow_steps: [prognostic_run]
  workflow_steps: [
    # coarsen_restarts,
    one_step_run,
    create_training_data,
    train_sklearn_model,
    # test_sklearn_model,
    prognostic_run,
  ]
  experiment_vars:
    one_step_yaml: /home/andrep/repos/fv3net/workflows/one_step_jobs/all-physics-off.yml
  steps:
    coarsen_restarts:
      command: workflows/coarsen_restarts/orchestrator_job.sh
      inputs:
        data_to_coarsen:
          location: gs://vcm-ml-data/orchestration-testing/shield-C384-restarts-2019-12-04
      method:
        source-resolution: 384
        target-resolution: 48
```