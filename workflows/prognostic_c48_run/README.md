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

The tests can be run with

	make test


Development
-----------

This workflow uses docker-compose to bootstrap a development environment. This
environment is based off the `prognostic_run` docker image, but has bind-mounts
to the packages in "/external" of this repository and this directory, which
allows locally developing this workflow and its dependencies. To enter a
development bash shell, run

    make dev


Configure the machine learning model (optional)
------------------------------------------

The ML model and location for zarr output can be configured using `fv3config.yml`. Add/modify the `scikit_learn` entry of the yaml file as follows (here to use a model from fv3fit.sklearn):
```
scikit_learn:
  model: gs://vcm-ml-data/test-annak/ml-pipeline-output
  model_type: scikit_learn
  zarr_output: diags.zarr
```
Or to use a model from fv3fit.keras:
```
scikit_learn:
  model: gs://vcm-ml-scratch/brianh/train-keras-model-testing/fv3fit-unified
  model_type: keras
  zarr_output: diags.zarr
```

Alternatively, the model path can also be specified via the 

If some variables names used for input variables in the scikit-learn model are inconsistent with the variable names used by the python wrapper, this can be handled by the optional `input_variable_standard_names` entry in the `scikit_learn` entry of the config yaml:
```
scikit_learn:
  input_variable_standard_names:
    DSWRFtoa_train: total_sky_downward_shortwave_flux_at_top_of_atmosphere
```

Prognostic Run in End-to-End Workflow
-------------------------------------

The prognostic run is included in the end-to-end workflow orchestration by `orchestrate_submit_job.py`. See `python orchestrate_submit_job.py -h` for current usage.

The prognostic run will grab the fv3config.yml file residing at the `initial_condition_url` argument and update it with any values specified in `prog_config_yml` argument, which can also include a `kubernetes`-key section to pass pod resource requests to `fv3run` (example shown below).  The prognostic-run ML configuration section "scikit_learn" is added (or updated if it already exists) to use the ML model specified as a command-line argument to `orchestrate_submit_job.py`.

The optional `diagnostics` entry in the prognostic run configuration can be used to save a subset of variables at set frequency or times.
If using this option to save the tendencies across the physics time step, also add a `physics_tendency_vars` entry to the `scikit_learn` configuration
option to specify these. An example is given below. If no `diagnostics` entry is included, the default behavior for the sklearn runfile
is to save the 2D diagnostics every 15 minutes to `diags.zarr`.

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
scikit_learn:
  physics_tendency_vars: 
    - air_temperature
    - specific_humidity
diagnostics:
  -
    name: physics_step_data.zarr
    times:
      frequency: 7200
      kind: interval
    variables:
      - tendency_of_air_temperature_due_to_fv3_physics
      - tendency_of_specific_humidity_due_to_fv3_physics

```
