# Usage

This workflow can be used to regrid cubed sphere FV3 data using GFDL's `fregrid` utility through [Kubeflow](https://kf-ml.endpoints.vcm-ml.cloud.goog/). In this webapp, you specify the input data (the prefix before `.tile?.nc`), the destination for the regridded outputs, and a comma separated list of variables to regrid from the source file.  Check out the Kubeflow pipeline [quickstart documentation](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/) for details of running a pipeline job.

Note: an image of this workflow is currently uploaded and usable under Kubeflow as the pipeline `regrid_individual_file`.  If the pipeline is not available please follow the process for building a docker image below.

## Building a docker image for Kubeflow

To build the docker image for this workflow and push it to GCR run
    
    make push
    
To deploy the pipeline, first install the kubeflow pipeline sdk:

    pip install https://storage.googleapis.com/ml-pipeline/release/latest/kfp.tar.gz


And then run
    
    make pipeline

This creates a tar file `regrid_c3072_diag.tar.gz`, that you can upload to [kubeflow](https://kf-ml.endpoints.vcm-ml.cloud.goog/) piplines web app. 

# Parameters

A description of user defined parameters used for the `regrid_individual_file` pipeline.

| Parameter | Description | Example |
|-----------|-------------|---------|
|`source_prefix`| Prefix of the source data in GCS (everything but .tile1.nc) | gs://path/to/sfc_data (no tile) |
| `output-bucket`| URL to output file in GCS | gs://vcm-ml-data/output.nc |
| `resolution`| Resolution of input data | one of 'C48', 'C96', or 'C384' |

# Miscellaneous

Kubeflow provides a REST API, that could be used to automate these steps. See
[this tutorial](https://www.suse.com/c/kubeflow-data-science-on-steroids/) for
an example of triggering a job via this API.
