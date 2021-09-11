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
