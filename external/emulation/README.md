emulation
=========

This is a stripped down set of modules with hooks for a prognostic run using `call_py_fort`.  It's currently used to create training datasets directly from data from the ZC microphysics parameterization as well as run online emulators.

The interaction points with python are located in `GFS_physics_driver.f90` as the `set_state`, `get_state`, or `call_function` commands.  These are enabled for the emulation by passing`CALLPYFORT=Y make -j8` to the fv3 compilation.

### Example snippet

```
#ifdef ENABLE_CALLPYFORT
            do k=1,levs
              do i=1,im
                qv_post_precpd(i,k) = Stateout%gq0(i,k,1)
                qc_post_precpd(i,k) = Stateout%gq0(i,k,ntcw)
              enddo
            enddo

            call set_state("air_temperature_output", Stateout%gt0)
            call set_state("specific_humidity_output", qv_post_precpd)
            call set_state("cloud_water_mixing_ratio_output", qc_post_precpd)
```

# Configuration

Emulator configuration, e.g., which model to load or the output formats to save, are controlled via environment variables.  It is convenient to usually set them in the runscript or create Argo parameters that map to these environment variables.

The current set of configurable environment variables are described in more detail in subsequent sections, but here's a brief list:

| Environment Variable     |    Description    |
| -------------------------|-------------------|
| OUTPUT_FREQ_SEC (int)| Frequency in seconds to save zarr files and/or netcdfs |
| SAVE_NC (bool) | Save netcdf files of the state from each rank at the specified output frequency |
| SAVE_ZARR (bool) | Save zarr files of the state at the specified output frequency |
| TF_MODEL_PATH (str) | Local/remote path to a tensorflow keras model to load |

## Training Data

**namelist parameter**
`gfs_physics_nml.save_zc_microphysics = True`

By default, the training data is saved out to the current working directory with a zarr monitor to state_output.zarr (time, tile, sample[y, x], z), or individual netCDF files for each time and tile under $(cwd)/netcdf_output.

To change the frequency for which data are saved (defaults to 5 hours [18,000 s]), prescribe the `OUTPUT_FREQ_SEC` environment variable in the runtime image.

To disable zarr or netCDF output, environment variables (`SAVE_NC`, `SAVE_ZARR`) can be set to `False`.

## Microphysics emulation

`gfs_physics_nml.emulate_zc_microphysics = True`

The microphysics emulation loads a keras model specified by the `TF_MODEL_PATH` environment variable.  Then during runtime, the model will make microphysical predictions for the ZC scheme on each rank.  

The model input/output names are used to update the state, so they should match the variables pushed into global state by call_py_fort.  See `GFS_physics_driver.F90` to see the list of available state variables.  Any state field that intersects with a field produced by the emulator will be adjusted to {state_name}_physics_diag so that piggy-backed information is present.

Running with `save_zc_microphysics = True` will save the emulator outputs and diagnostic physics info to file.


## Loading TFRecord data

TFRecord is a binary file format that it native to tensorflow. This means that
loading it requires no pure-python code and should be portable and performant.
If `SAVE_TFRECORD` is set, then the model will save each rank to a separate
`.tfrecord` file. Because tfrecords contain raw serialized data, we also save a
tf module to `parser.tf` to parse it. To open the data, use the
following boilerplate:

```python
url = "gs://vcm-ml-experiments/microphysics-emulation/2021-12-13/pzp6alfw/artifacts/20160611.000000/tfrecords"
parser = tf.saved_model.load(f"{url}/parser.tf")
tf_ds = tf.data.TFRecordDataset(
    tf.data.Dataset.list_files(f"{url}/*.tfrecord")
).map(parser.parse_single_example)
```

Note that tensorflow I/O routines support `GCS` links beginning with
`gs://some-bucket/some-path`.
