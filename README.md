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
    ├── workflows           <-- Job submission scripts and description for pieces of data pipeline
    │   ├── extract_tars        <-- Dig yourself out of a tarpit using Dataflow
    │   ├── rerun-fv3           <-- Perform a single step run of FV3 for available timesteps
    │   └── scale-snakemake     <-- Coarsening operation with kubernetes and snakemake (Deprecated)
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

To install requirements for development and running of this package, run

    pip install -r requirements.txt


To make sure that `fv3net` is in the python path, you can run

    python setup.py develop

or possible add it to the `PYTHONPATH` environmental variable.

## Adding VCM Tools

Currently VCM Tools , which is used by the pipeline resides within this repository 
in the external folder.  This will be eventually spun off into its own repo.  To 
make sure that `vcm` is in the Python path, you can run

    $ cd external/vcm
    $ python setup.py install    


# Deploying cloud data pipelines

The main data processing pipelines for this project currently utilize Google Cloud
Dataflow and Kubernetes with Docker images.  Run scripts to deploy these workflows
along with information can be found under the `workflows` directory.

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


## Deploying on k8s  (likely outdated?)

Make docker image for this workflow and push it to GCR

    make push_image

Create a K8S cluster:

    bash provision_cluster.sh

This cluster has some big-mem nodes for doing the FV3 run, which requires at least a n1-standard-2 VM for 
C48.

Install argo following [these instructions](https://github.com/argoproj/argo/blob/master/demo.md).

Submit an argo job using

    argo submit --watch argo-fv3net.yml


# Extending this code

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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
