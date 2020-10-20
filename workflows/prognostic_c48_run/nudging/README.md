## Nudging workflow

This workflow performs a nudged run, nudging to reference data stored on GCS.
It is included as part of the prognostic_c48_run workflow as a WIP while we
consolidate workflows performing different flavours of fv3gfs running.


### Configuration

As with the prognostic run, the nudging run is configured
by specifying an update to the base configurations in `fv3kube`. The runfile
requires a `nudging` section within the fv3config object. This section
contains the location of the nudging dataset as well as the nudging
time-scales. Here is an example:
```
base_version: v0.4
forcing: gs://vcm-fv3config/data/base_forcing/v1.1/
initial_conditions: /mnt/input/coarsen_restarts/20160801.001500/
nudging:
  restarts_path: /mnt/input/coarsen_restarts/
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