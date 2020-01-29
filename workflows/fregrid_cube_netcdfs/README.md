# Usage

This workflow can be used to regrid cubed sphere FV3 data using GFDL's `fregrid` utility using [argo][1]. In this workflow, you specify the input data (the prefix before `.tile?.nc`), the destination for the regridded outputs, and a comma separated list of variables to regrid from the source file.

Note: an image of this workflow is currently uploaded and usable under Kubeflow as the pipeline `regrid_individual_file`.  If the pipeline is not available please follow the process for building a docker image below.

## Building a docker image for Kubeflow

To build the docker image for this workflow and push it to GCR run
    
    make push
    
Assuming kubectl is configured correctly, and the argo CLI is installed locally, a job can be run like this

    argo submit pipeline.yaml -p source_prefix=<url to data>  [any other args with -p]

The other parameters describe below can be passed with the -p flags.

# Parameters

A description of user defined parameters used for the `regrid_individual_file` pipeline.

| Parameter | Description | Example |
|-----------|-------------|---------|
|`source_prefix`| Prefix of the source data in GCS (everything but .tile1.nc) | gs://path/to/sfc_data (no tile) |
| `output-bucket`| URL to output file in GCS | gs://vcm-ml-data/output.nc |
| `resolution`| Resolution of input data | one of 'C48', 'C96', or 'C384' |
| `--extra_args`| Extra arguments to pass to fregrid. Typically used to specify the target resolution | --nlat 180 --nlon 360 |

[1]: https://github.com/argoproj/argo
