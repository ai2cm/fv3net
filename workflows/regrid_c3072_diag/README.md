# Usage

To build the docker image for this workflow and push it to GCR run
    
    make push
    
To deploy the pipeline, first install the kubeflow pipeline sdk:

    pip install https://storage.googleapis.com/ml-pipeline/release/latest/kfp.tar.gz


And then run
    
    make pipeline

This creates a tar file `regrid_c3072_diag.tar.gz`, that you then upload can upload to [kubeflow](https://kf-ml.endpoints.vcm-ml.cloud.goog/) piplines web app. In this webapp, you specific the input data (the prefix before `.tile?.nc`), the destination for the regridded outputs, and a comma separated list of variables to regrid from the source file.

# Parameters

| Parameter | Description | Example |
|`source_prefix`| Prefix of the source data in GCS (everything but .tile1.nc) | gs://path/to/sfc_data (no tile) |
| `output-bucket`| URL to output file in GCS | gs://vcm-ml-data/output.nc |
| `resolution`| Resolution of input data | one of 'C48', 'C96', or 'C384' |
