Workflow to perform many short FV3GFS runs initialized from sequential timesteps.

Specific model configurations can be specified through the `one-step-yaml` argument.

Included configurations are:
- `all-physics-off.yml` (model will not do any physics or
microphysics)
- `deep-and-mp-off.yml` (model will not do microphysics or deep convection).

Both of these configurations use a one-minute timestep with no dynamics substepping and
have a total duration of 15 minutes.

To process many (> around 40) runs at once, it is recommended to submit this workflow
from a VM authorized with a service account. Users have had issues with API request errors
when submitting from a machine authorized with a non-service Google account.

```
$ python submit_jobs.py -h
usage: submit_jobs.py [-h] --one-step-yaml ONE_STEP_YAML --input-bucket
                      INPUT_BUCKET --output-bucket OUTPUT_BUCKET
                      [--n-steps N_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --one-step-yaml ONE_STEP_YAML
                        Path to local run configuration yaml.
  --input-bucket INPUT_BUCKET
                        Remote url to initial conditions. Initial conditions
                        are assumed to be stored as INPUT-BUCKET/{timestamp}/
                        {timestamp}.{restart_category}.tile*.nc
  --output-bucket OUTPUT_BUCKET
                        Remote url where model configuration and output will
                        be saved. Specifically, configuration files will be
                        saved to OUTPUT-BUCKET/config and model output to
                        OUTPUT-BUCKET/rundirs
  --n-steps N_STEPS     Number of timesteps to process. By default all
                        timesteps found in INPUT-BUCKET will be processed.
                        Useful for testing.

```