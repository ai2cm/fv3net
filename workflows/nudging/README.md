## Nudging workflow

This workflow performs a nudged run, nudging to reference data stored on GCS.

### Quickstart

Pull the docker image from GCS, if you don't have it already:

    docker pull us.gcr.io/vcm-ml/fv3gfs-python:0.4.1

Run the workflow:

    make run

The output directory should now be present in `output`.

### Running on Kubernetes

The workflow can be submitted to kubernetes using:

    make run_kubernetes

You should probably specify the remote output directory to use, as follows:

    REMOTE_ROOT=gs://my-bucket/ make run_kubernetes

### Configuration

The reference restart location and variables to nudge are stored in `fv3config_base.yml`.
The nudging timescales in that file are ignored, and instead replaced with the
TIMESCALE_HOURS variable set in the makefile (which you can pass manually).

### More details

The nudging and run are configured in `fv3config_base.yml`. This gets converted into
`fv3config.yml` automatically in order to specify the full filenames for initial
conditions on GCS that has been prepended with the timestamp.

If you want to run the workflow on a different image, you can set `IMG_NAME` and `IMG_VERSION` when you call `make`.
