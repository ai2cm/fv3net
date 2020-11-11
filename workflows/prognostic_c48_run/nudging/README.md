## Nudging workflow

This workflow performs a nudged run, nudging to reference data stored on GCS.
It is included as part of the prognostic_c48_run workflow as a WIP while we
consolidate workflows performing different flavours of fv3gfs running.


### Configuration

As with the prognostic run, the nudging run is configured
by specifying an update to the base configurations in `fv3kube`. Nudging
requires a `nudging` section within the fv3config object. This section
contains the location of the nudging dataset as well as the nudging
time-scales. Here is an example:
```
base_version: v0.5
nudging:
  timescale_hours:
    air_temperature: 3
    specific_humidity: 3
    x_wind: 3
    y_wind: 3
    pressure_thickness_of_atmospheric_layer: 3
  reference_initial_time: "20160801.001500"
  reference_frequency_seconds: 900
namelist: {}
```

This runfile supports nudging towards a dataset with a different sampling
frequency than the model time step. The available nudging times should start
with `initial_time` and appear at a regular frequency of `frequency_seconds`
thereafter. These options are optional, if not provided the nudging data will
be assumed to contain every time. The reference state will be linearly
interpolated between the available time samples.


### Running locally

A nudged run may be run locally within the prognostic_run image, which can be
entered following the instructions in `workflows/prognostic_c48_run/README.md`.
Once in the container, run `make` from this directory to launch an example run using
`./nudging_config.yaml`.


### Running with argo

An argo workflow template is provided to launch a nudging run. See the README in
`workflows/argo` for more information.
