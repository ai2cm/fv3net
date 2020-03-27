fv3net
==============================
[![CircleCI](https://circleci.com/gh/VulcanClimateModeling/fv3net.svg?style=svg&circle-token=98ccddae8375060a2fbbf240407dd4135d3dcf68)](https://circleci.com/gh/VulcanClimateModeling/fv3net)

Improving the GFDL FV3 model physics with machine learning

Project Organization
------------

    ├── assets              <-- Useful files for restart directory creation? 
    ├── configurations      <-- Files for ML model configuration and general configuration
    ├── data                <-- Intermediate data store
    ├── docker 
    ├── external            <-- Package dependencies that will be spun off into their own repo
    │   └── vcm                 <-- General VCM tools package 
    ├── fv3net              <-- Main package source for ML pipelines and model code
    │   ├── models
    │   ├── pipelines           <-- Cloud data pipelines
    │   ├── visualization
    ├── tests
    ├── workflows                     <-- Job submission scripts and description for pieces of data pipeline
    │   ├── coarsen_c384_diagnostics  <-- Coarsen FV3GFS C384 diagnostic files
    │   ├── coarsen_restarts          <-- Coarsen FV3GFS restart timesteps to desired resolution
    │   ├── create_training_data      <-- Create training data batches from Fv3GFS step simulations
    │   ├── extract_tars              <-- Dig yourself out of a tarpit using Dataflow
    │   ├── end_to_end                <-- Run end-to-end pre-processing, training, and prognostic run
    │   ├── one_step_jobs             <-- Submit one-step kubernetes jobs using fv3run
    │   ├── prognostic_c48_run        <-- Run a prognostic Fv3GFS simulation with an embedded ML model
    │   ├── single_fv3gfs_run         <-- Submit a one off free or nudged fv3gfs simulation
    │   ├── fregrid_cube_netcdfs      <-- Regrid cubed-sphere to lat/lon and other grids data with fregrid
    │   ├── scale-snakemake           <-- Coarsening operation with kubernetes and snakemake (Deprecated)
    │   └── sklearn_regression        <-- Train an ML model
    ├── Dockerfile
    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── catalog.yml         <-- Intake list of datasets 
    ├── environment.yml
    ├── pytest.ini
    ├── regression_tests.sh
    ├── requirements.txt
    └── setup.py

--------

# Setting up the environment

This computational environment can be challenging to install because it require
both external packages as well as tools developed locally at Vulcan. The
internal Vulcan dependencies are included as submodules in the `external`
folder, while the external dependencies are managed using anaconda with an
`environment.yml`. The Vulcan submodules can be download, if they aren't
already, by running

    make update_submodules

Then, assuming anaconda is installed, the environment can be created by running

    make create_environment

This creates an anaconda environment `fv3net` containing both Vulcan and
external dependencies. It downloads all the Vulcan submodules and installs them
in development mode (e.g. `pip install -e`) so that any modifications within
`external` will be loaded in the conda environment. Once this command completes
succesfully, the fv3net environment can be activated with

    conda activate fv3net

The `build_environment.sh` script also outputs a list of all installed dependencies
with their version under `.circleci/.installed_build_env_deps`.  This file is used
along with `environment.yml` as a key for caching the `fv3net` dependencies.  Whenver
`make create_environment` or the build script is run, this file will be updated
and committed to keep track of versions over time.

# Deploying cloud data pipelines

The main data processing pipelines for this project currently utilize Google Cloud
Dataflow and Kubernetes with Docker images.  Run scripts to deploy these workflows
along with information can be found under the `workflows` directory.

## Building the fv3net docker images

The workflows use a pair of common images:

|Image| Description| 
|-----|------------|
| `us.gcr.io/vcm-ml/prognostic_run` | fv3gfs-python with minimal fv3net and vcm installed |
| `us.gcr.io/vcm-ml/fv3net` | fv3net image with all dependencies including plotting |

These images can be built and pushed to GCR using `make build_images` and
`make push_images`, respectively.

## Dataflow

Dataflow jobs run in a "serverless" style where data is piped between workers who
execute a single function.  This workflow type is good for pure python operations
that are easily parallelizable.  Dataflow handles resource provision and worker
scaling making it a somewhat straightforward option.

For a dataflow pipeline, jobs can be tested locally using a DirectRunner, e.g., 

    python -m fv3net.pipelines.extract_tars test_tars test_output_dir --runner DirectRunner

To submit a Dataflow job to the cloud involves a few steps, including packaging 
our external package `vcm` to be uploaded.

    $ cd external/vcm
    $ python setup.py sdist


After creating the uploadable `vcm` package, submit the Dataflow job from the top 
level of `fv3net` using:

    python -m fv3net.pipelines.extract_tars \
        test_tars \
        test_outputs \
        --job_name test-job \   
        --project vcm-ml \
        --region us-central1 \
        --runner DataflowRunner \
        --temp_location gs://vcm-ml-data/tmp_dataflow \
        --num_workers 4 \
        --max_num_workers 20 \
        --disk_size_gb 50 \
        --type_check_strictness 'ALL_REQUIRED' \
        --worker_machine_type n1-standard-1 \
        --setup_file ./setup.py \
        --extra_package external/vcm/dist/vcm-0.1.0.tar.gz

We provide configurable job submission scripts under workflows to expedite this process. E.g.,

    workflows/extract_tars/submit_job.sh

If you get an error `Could not create workflow; user does not have write access to project` upon
trying to submit the dataflow job, do `gcloud auth application-default login` first and then retry.



## Deploying on k8s with fv3net

Docker images with the python-wrapped model and fv3run are available from the
[fv3gfs-python](https://github.com/VulcanClimateModeling/fv3gfs-python) repo.
Kubernetes jobs can be written to run the model on these docker images. A super simple
job would be to perform an `fv3run` command (provided by the
[fv3config package](https://github.com/VulcanClimateModeling/fv3config))
using google cloud storage locations. For example, running the basic model using a
fv3config dictionary in a yaml file to output to a google cloud storage bucket
would look like:

```
fv3run gs://my_bucket/my_config.yml gs://my_bucket/my_outdir
```

If you have a python model runfile you want to execute in place of the default model
script, you could use it by adding e.g. `--runfile gs://my-bucket/my_runfile.py`
to the `fv3run` command.

You could create a kubernetes yaml file which runs such a command on a
`fv3gfs-python` docker image, and submit it manually. However, `fv3config` also
supplies a `run_kubernetes` function to do this for you. See the
[`fv3config`](https://github.com/VulcanClimateModeling/fv3config) documentation for
more complete details, or the `one_step_jobs` workflow for a more complex example of
using the function to prepare and submit many jobs.

The basic structure of the command is

    fv3config.run_kubernetes(
        config_location,
        outdir,
        docker_image,
        gcp_secret='gcp_key',
    )

Where `config_location` is a google cloud storage location of a yaml file containing
a fv3config dictionary, outdir is a google cloud storage location to put the resulting
run directory, `docker_image` is the name of a docker image containing `fv3config`
and `fv3gfs-python`, and `gcp_secret` is the name of the secret containing the google
cloud platform access key (as a json file called `key.json`). For our VCM group this
should be set to 'gcp_key'. Additional arguments are
available for configuring the kubernetes job and documented in the `run_kubernetes`
docstring.

## Adding new model types

*This interface is a work in progress. It might be better to define a class
based interface. Let's defer that to when we have more than one model type*

To add a new model type one needs to create two new files
```
fv3net.models.{model_type}/train.py
fv3net.models.{model_type}/test.py
```

The training script should take in a yaml file of options with this command line interface
```
python -m fv3net.models.{model_type}.train --options options.yaml output.pkl
```

# Code linting checks

This python code in this project is autoformated using the
[black](https://black.readthedocs.io/en/stable/) code formatting tool, and the
[isort](https://github.com/timothycrosley/isort) tool for automatically sorting
the order of import statements. To pass CI, any contributed code must be
unchanged by black and also checked by the flake8 linter. However, please use
isort to sort the import statements (done automatically by `make reformat`
below).

Contributers can see if their *commited* code passes these standards by running

    make lint

If it does not pass, than it can be autoformatted using 

    make reformat

# How to contribute to fv3net

Please see the [contribution guide.](./CONTRIBUTING.md)

# How to get updates and releases

For details on what's included or upcoming for a release, please see the [HISTORY.rst](./HISTORY.rst) document.

For instructions on preparing a release, please read [RELEASE.rst](./RELEASE.rst).

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>