# Regrid ERA5 2D reanalysis variables

This folder contains resources to quickly regrid ERA5 2D reanalysis
variables used for the ocean reservoir model.  Currently it uses
dataflow (with xarray-beam) to chunk the data into individual datasets
that are regridded to 1deg regular lat/lon grids from the native
reduced gaussian grid.

## Usage

To deploy the workflow you'll likely need to create a new conda environment
with updated packages for interacting with Dataflow:

    make dataflow_conda_env

Then you can deploy the workflow using:

    make deploy


## Testing

To test the dataflow pipeline locally, you can use the following command:

    make test_local

This utilizes the `DirectRunner` to run the pipeline locally within the
docker image.  This is useful for debugging the pipeline.

If you'd just like to work interactively within the docker image, you can
run:

    make enter

## Docker image

The docker image contains all the necessary python packages and regridding
dependencies (i.e., CDO). The docker image is already built and pushed to
GCR for this purpose.  Should you need to recreate the image, you can run:

    make build_docker

or to build and push:

    make push_docker

