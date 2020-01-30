This workflow provides a re-usable job submission script for doing long one-off free
or nudged simulations with the FV3GFS model. Example run configurations for 1-year long
simulations are `long_free.yml` and `long_nudged.yml` for free-running and nudged respectively.
Run configuration include both information about the job (# processors, memory) as well as a
standard fv3config model configuration dictionary. Only non-default parameters need to be
specified. Feel free to add more run configurations to this folder if you develop some.

Jobs may be submitted by calling `submit_job.py`:
```
usage: submit_job.py [-h] --bucket BUCKET --run-yaml RUN_YAML

optional arguments:
  -h, --help           show this help message and exit
  --bucket BUCKET      Remote url where config and output will be saved.
                       Specifically, configuration will be saved to
                       BUCKET/config and output to BUCKET/output
  --run-yaml RUN_YAML  Path to local run configuration yaml.
```


