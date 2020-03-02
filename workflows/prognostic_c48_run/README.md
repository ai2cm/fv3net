Prognostic model run workflow
=============================

This workflow makes makes a coupled C48 FV3 run with a scikit-learn model. This workflow mainly
1. builds a docker image suitable for online prognostic runs with fv3gfs, and
1. includes runfile `sklearn_runfile.py` which can be used with `fv3run`

(Authenticating) with google cloud
--------------------------------

This workflow needs to access google cloud storage within docker. To do this,
it expects that the environmental variable GOOGLE_APPLICATION_CREDENTIALS
will point to a valid service account key on the host system. For example,

    export GOOGLE_APPLICATION_CREDENTIALS=<path/to/key.json>

Quickstart
----------

To build the docker image to used, run

	make build

The git revision of the fv3net installed in the docker image can be specified using the `FV3NET_VERSION` flag. For example,

    FV3NET_VERSION=master make build
    
will install the master branch of fv3net.

To start a bash shell in this image, run

	make dev

An example prognostic run can be started with 

	make sklearn_run
    
Saving the fv3gfs-python state as a pickle
------------------------------------------

For prototyping purposes it is useful to be able to save the state as returned
by `fv3gfs.get_state` so it can be loaded outside the docker image. This can be
done by running

	make state.pkl

This will create a file `state.pkl`, which can be read in python using 
	
```python
import state_io

with open("state.pkl", "rb") as f:
    data = state_io.load(f)
```

The main method of the  `online_modules/sklearn_interface` reads in this file and prints some outputs.

    make test_run_sklearn

Note that the `online_modules` must be in the PYTHONPATH for this make rule to work.

Configure the scikit-learn run
------------------------------------------

The scikit-learn model and location for zarr output can be configured using `fv3config.yml`. To do this, simply add/modify the `scikit_learn` entry of the yaml file as follows:
```
scikit_learn:
  model: gs://vcm-ml-data/test-annak/ml-pipeline-output/2020-01-17_rf_40d_run.pkl
  zarr_output: diags.zarr
```

Prognostic Run in End-to-End Workflow
-------------------------------------

The prognostic run is included in the end-to-end workflow orchestration by `orchestrate_submit_job.py`.  This script takes command-line arguments:

```
usage: orchestrate_submit_job.py [-h]
                                 model_url initial_condition_url output_url
                                 prog_config_yml ic_timestep docker_image

positional arguments:
  model_url             Remote url to a trained sklearn model.
  initial_condition_url
                        Remote url to directory holding timesteps with model
                        initial conditions.
  output_url            Remote storage location for prognostic run output.
  prog_config_yml       Path to a config update YAML file specifying the
                        changes (e.g., diag_table, runtime, ...) from the one-
                        step runs for the prognostic run.
  ic_timestep           Time step to grab from the initial conditions url.
  docker_image          Docker image to pull for the prognostic run kubernetes
                        pod.
```

The prognostic run will grab the fv3config.yml file residing at the `initial_condition_url` and update it with any values specified in `prog_config_yml`, which can also include a `kubernetes`-key section to pass pod resource requests to `fv3run` (example shown below).  A prognostic-run configuration section ("scikit_learn") is added to the configuration for execution of the ML model within this workflow member.

### `prog_config_yml` example

```yaml
kubernetes:
  cpu_count: 6
  memory_gb: 3.6
diag_table: workflows/prognostic_c48_run/diag_table_prognostic
namelist:
  coupler_nml:
    hours: 0
    minutes: 60
    seconds: 0
```
