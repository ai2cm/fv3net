```
usage: submit_job.py [-h] --bucket BUCKET --run-yaml RUN_YAML

optional arguments:
  -h, --help           show this help message and exit
  --bucket BUCKET      Remote url where config and output will be saved.
                       Specifically, configuration will be saved to
                       BUCKET/config and output to BUCKET/output
  --run-yaml RUN_YAML  Path to local run configuration yaml.
  --config-version CONFIG_VERSION
                        Default fv3config.yml version to use as the base
                        configuration. This should be consistent with the
                        fv3gfs-python version in the specified docker image.
                        Defaults to fv3gfs-python v0.2 style configuration.
```
This workflow provides a re-usable job submission script for doing long one-off free
or nudged simulations with the FV3GFS model. Example run configurations for 1-year long
simulations are `long_free.yml` and `long_nudged.yml` for free-running and nudged respectively.
Run configuration include both information about the job (# processors, memory, runfile) as well as a
standard fv3config model configuration dictionary. Only non-default parameters need to be
specified. Feel free to add more run configurations to this folder if you develop some.

A custom `runfile` can be specified in the `kubernetes` section of the run_yaml. The provided
examples do not specify the `runfile`, and hence use the default.

Note that all configuration assets except the `diag_table` and `runfile` are assumed to already
be on GCS. This includes any initial conditions or patch file assets.

The below table summarizes the configurations available. It has a good chance of being out of date.

| config | start date | length | nudged vars | default repo | SST |
|--------|------------|--------|-------------|--------------|-----|
| long_free.yml | 1 Jan 2016 | 1 year | none | fv3gfs | clim |
| long_nudged.yml | 1 Jan 2016 | 1 year | T, ps, u, v | fv3gfs | clim |
| 40day_free.yml | 1 Aug 2016 | 40 days | none | fv3atm | GFS analysis |
| 40day_nudged.yml | 1 Aug 2016 | 40 days | T, ps, u, v | fv3atm | GFS analysis |
| free_GFS_SST_2015.yml | 1 Jan 2015 | 1 year | none | fv3atm | GFS analysis |
| nudge_T_2015.yml | 1 Jan 2015 | 1 year | T | fv3atm | GFS analysis |
| nudge_T_ps_2015.yml | 1 Jan 2015 | 1 year | T, ps | fv3atm | GFS analysis |
| nudge_T_ps_u_v_2015.yml | 1 Jan 2015 | 1 year | T, ps, u, v | fv3atm | GFS analysis |
