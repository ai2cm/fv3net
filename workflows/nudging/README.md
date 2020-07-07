## Nudging workflow

This workflow performs a nudged run, nudging to reference data stored on GCS.


### Configuration

As with the prognostic run and one-steps runs, the nudging run is configured
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
namelist: {}
```


### Local Development

Pull the docker image from GCS, if you don't have it already:

    docker pull us.gcr.io/vcm-ml/fv3gfs-python:0.4.1

Run the workflow:

    make run

The output directory should now be present in `output`.

If you want to run the workflow on a different image, you can set `IMG_NAME`
and `IMG_VERSION` when you call `make`.


### Running with argo


Argo expects to be passed the contents of nudging configuration as a string.
This can either be done using `argo submit -f <argo config>` where `<argo
config>` is similar to `examples/argo_clouds_off.yaml`. Or, you can use the
`-p` flag of argo submit:

    argo submit -p output-url=gs://path -p nudging-config="$(cat nudging_config.yaml)"

See the `argo.yaml` file for the available workflow parameters.
