Nudging workflow
================

This workflow performs a nudged run, nudging to reference data stored on GCS.

Quickstart
----------

Pull the docker image from GCS:

    docker pull us.gcr.io/vcm-ml/fv3gfs-python:0.3.1-nudging

Run the workflow:

    make run

The output directory should now be present in `output`.

Building the image
------------------

Build requires an `fv3gfs-python` base image. If you don't have one with the
correct version tag, it will be automatically pulled by `make build`. It then produces an image
tagged with `-nudging` which contains the cached data needed to perform the nudging run,
and any additional python packages needed.

Building the `-nudging` image requires `GOOGLE_APPLICATION_CREDENTIALS` to be set.
These credentials are copied into an intermediate image to download the data cache.
The credentials are not included in the final `-nudging` image.

More details
------------

The nudging and run are configured in `fv3config_base.yml`. This gets converted into
`fv3config.yml` automatically in order to specify the full filenames for initial
conditions on GCS that has been prepended with the timestamp.

`retry.sh` performs a rebuild of the `fv3gfs-python` base image, a rebuild of the
nudging image, and then repeats the workflow, showing the error log if an error
occurs. It requires an environment variable `FV3GFS_PYTHON_DIR` which points to
the directory where the `fv3gfs-python` repo is stored. It assumes that the repo is
checked out to the correct version of `fv3gfs-python`.
