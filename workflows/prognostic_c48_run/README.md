Prognostic model run workflow
=============================

This workflow makes a coupled C48 FV3 run with a scikit-learn model. This workflow
1. includes runfile `sklearn_runfile.py` which can be used with `fv3run`
1. uses the `prognostic_run` docker image, built by `fv3net/docker/prognostic_run/Dockerfile`

(Authenticating) with google cloud
--------------------------------

This workflow needs to access google cloud storage within docker. To do this,
it expects that the environmental variable GOOGLE_APPLICATION_CREDENTIALS
will point to a valid service account key on the host system. For example,

    export GOOGLE_APPLICATION_CREDENTIALS=<path/to/key.json>

Quickstart
----------

An example prognostic run can be started with

	make sklearn_run

Configure the scikit-learn run
------------------------------------------

The scikit-learn model and location for zarr output can be configured using `fv3config.yml`. To do this, simply add/modify the `scikit_learn` entry of the yaml file as follows:
```
scikit_learn:
  model: gs://vcm-ml-data/test-annak/ml-pipeline-output/2020-01-17_rf_40d_run.pkl
  zarr_output: diags.zarr
```

If some variables names used for input variables in the scikit-learn model are inconsistent with the variable names used by the python wrapper, this can be handled by the optional `input_variable_standard_names` entry in the `scikit_learn` entry of the config yaml:
```
scikit_learn:
  input_variable_standard_names:
    DSWRFtoa_train: total_sky_downward_shortwave_flux_at_top_of_atmosphere
```

Prognostic Run in End-to-End Workflow
-------------------------------------

The prognostic run is included in the end-to-end workflow orchestration by `orchestrate_submit_job.py`.  This script takes command-line arguments:

```
usage: orchestrate_submit_job.py [-h] [--model_url MODEL_URL]
                                 [--prog_config_yml PROG_CONFIG_YML]
                                 [--diagnostic_ml] [-d]
                                 initial_condition_url ic_timestep
                                 docker_image output_url

positional arguments:
  initial_condition_url
                        Remote url to directory holding timesteps with model
                        initial conditions.
  ic_timestep           Time step to grab from the initial conditions url.
  docker_image          Docker image to pull for the prognostic run kubernetes
                        pod.
  output_url            Remote storage location for prognostic run output.

optional arguments:
  -h, --help            show this help message and exit
  --model_url MODEL_URL
                        Remote url to a trained sklearn model.
  --prog_config_yml PROG_CONFIG_YML
                        Path to a config update YAML file specifying the
                        changes from the basefv3config (e.g. diag_table,
                        runtime, ...) for the prognostic run.
  --diagnostic_ml       Compute and save ML predictions but do not apply them
                        to model state.
  -d, --detach          Do not wait for the k8s job to complete.
```

The prognostic run will grab the fv3config.yml file residing at the `initial_condition_url` and update it with any values specified in `prog_config_yml`, which can also include a `kubernetes`-key section to pass pod resource requests to `fv3run` (example shown below).  The prognostic-run ML configuration section "scikit_learn" is added (or updated if it already exists) to use the ML model specified as a command-line argument to `orchestrate_submit_job.py`.

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
