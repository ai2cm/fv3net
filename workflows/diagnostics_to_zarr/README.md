## Diagnostics-to-zarr workflow
This workflow takes a path/url to a run directory as an input and saves zarr stores
of the diagnostic model output to a specified location. This workflow requires a 
specific xarray version (0.14.0) and so to run locally, one must ensure your 
environment is using that version. For dataflow jobs, a custom setup.py is provided. 

The example scripts `submit_local.sh` and `submit_job.sh` are provided for local
and Dataflow jobs, respectively. 

```
usage: diagnostics_to_zarr.py [-h] --rundir RUNDIR
                              [--diagnostic-dir DIAGNOSTIC_DIR]
                              [--diagnostic-categories DIAGNOSTIC_CATEGORIES [DIAGNOSTIC_CATEGORIES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --rundir RUNDIR       Location of run directory. May be local or remote
                        path.
  --diagnostic-dir DIAGNOSTIC_DIR
                        Location to save zarr stores. Defaults to the parent
                        of rundir.
  --diagnostic-categories DIAGNOSTIC_CATEGORIES [DIAGNOSTIC_CATEGORIES ...]
                        Optionally specify one or more categories of
                        diagnostic files. Provide part of filename before
                        .tile*.nc. Defaults to all diagnostic categories in
                        rundir.
```
