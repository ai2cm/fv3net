Prognostic run output data specs
--------------------------------

Data specifications for the sklearn
runfile for nudge-to-fine, nudge-to-obs, and ML/baseline runs, which
correponds to data expectations of the nudge-to-fine and nudge-to-obs
mappers and the prognostic run report generation workflow step.

Summary of data outputs
~~~~~~~~~~~~~~~~~~~~~~~

The following summarizes data collections which will be output from the
run depending on the configuration type chosen, focusing on those
outputs used by data mappers for training (nudge-to-x runs) or in
generating the prognostic run report. Additional Fortran diagnostic
collections (e.g., atmos_dt_atmos) are specified by the diag_table file.

+-----------------+------------------------------+------------------------------+-----------------+
| Output Type     | Nudge-to-fine                | Nudge-to-obs                 | Prognostic (ML  |
|                 | (N2F)                        | (N2O)                        | and baseline)   |
+=================+==============================+==============================+=================+
| 3-D physics     | ``physics_tendencies.zarr``  | ``physics_tendencies.zarr``  | N/A             |
| tendencies      |                              |                              |                 |
| (pQs)           |                              |                              |                 |
+-----------------+------------------------------+------------------------------+-----------------+
| 3-D nudging     | ``nudging_tendencies.zarr``  | ``nudging_tendencies.zarr``  | N/A             |
| tendencies      |                              | (Fortran                     |                 |
| (dQs)           |                              | diagnostic                   |                 |
|                 |                              | zarr)                        |                 |
+-----------------+------------------------------+------------------------------+-----------------+
| 3-D model state | ``state_after_timestep.zarr``| ``state_after_timestep.zarr``| N/A             |
+-----------------+------------------------------+------------------------------+-----------------+
| 2-D diagnostics | ``diags.zarr``               | N/A                          | ``diags.zarr``  |
| (for prognostic |                              |                              |                 |
| report, not     |                              |                              |                 |
| mappers)        |                              |                              |                 |
+-----------------+------------------------------+------------------------------+-----------------+

-  **physics_tendencies.zarr**: 3-D physics tendencies computed via
   sklearn_runfile monitor function, e.g.,
   ``tendency_of_air_temperature_due_to_fv3_physics`` and
   ``tendency_of_eastward_wind_due_to_fv3_physics``; note that for
   nudge-to-obs only this is output as the sum of physics and nudging
   tendencies (actual physics tendencies are computed by the
   nudge-to-obs mapper)

-  **nudging_tendencies.zarr** (N2F): 3-D nudging tendencies computed
   via sklearn_runfile nudge-to-fine capability, e.g.,
   ``air_temperature_tendency_due_to_nudging``

-  **nudge_to_obs_tendencies.zarr** (N2O): 3-D nudging tendencies
   computed via Fortran diagnostic outputs post-processed to zarr, e.g.,
   ``u_dt_nudge``

-  **state_after_timestep.zarr**: model state storage containing longer
   list of variables suitable for ML features and diagnosis. Output from
   state at end of timestep for consistency with Fortran diagnostics, so
   “pre-nudging” state is then computed by mapper from the nudging
   tendencies. Candidate list based on outputs from previous
   nudge-to-fine and nudge to obs runfiles:

   -  x_wind
   -  y_wind
   -  eastward_wind
   -  northward_wind
   -  vertical_wind
   -  air_temperature
   -  specific_humidity
   -  time
   -  pressure_thickness_of_atmospheric_layer
   -  vertical_thickness_of_atmospheric_layer
   -  land_sea_mask
   -  surface_temperature
   -  surface_geopotential
   -  sensible_heat_flux
   -  latent_heat_flux
   -  total_precipitation
   -  surface_precipitation_rate
   -  total_soil_moisture
   -  total_sky_downward_shortwave_flux_at_surface
   -  total_sky_upward_shortwave_flux_at_surface
   -  total_sky_downward_longwave_flux_at_surface
   -  total_sky_upward_longwave_flux_at_surface
   -  total_sky_downward_shortwave_flux_at_top_of_atmosphere
   -  total_sky_upward_shortwave_flux_at_top_of_atmosphere
   -  total_sky_upward_longwave_flux_at_top_of_atmosphere
   -  clear_sky_downward_shortwave_flux_at_surface
   -  clear_sky_upward_shortwave_flux_at_surface
   -  clear_sky_downward_longwave_flux_at_surface
   -  clear_sky_upward_longwave_flux_at_surface
   -  clear_sky_upward_shortwave_flux_at_top_of_atmosphere
   -  clear_sky_upward_longwave_flux_at_top_of_atmosphere
   -  latitude
   -  longitude

-  **diags.zarr** (not read by mapper but incorporates nudge-to-fine and
   ML diags into prognostic run report): 2-D diagnostics computed by the
   python routine, e.g. for nudging ``net_moistening_due_to_nudging``,
   ``column_integrated_northward_wind_tendency_due_to_nudging``
