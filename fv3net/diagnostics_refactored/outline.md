### Refactored offline diagnostics workflow
Other concerns
- do we want to continue to create figures like the "diurnal cycle across workflow", or was that a one off for the presentation
- OOM issues when loading data
   - `.load()` only necessary values
   - make sure the grid isn't getting a time dim
- is there a significant cost increase for copying data to the local drive, then reading from there, then delete after use? this is much faster that loading from GCS and we don't have to worry about retrying upon read errors
- just hardcode the x/y/tile coords for the particular location points in the constants rather than doing the lat/lon lookup each time this runs.
 
 
 
contents of `fv3net.diagnostics`
- time avg map plot of dQ1/2, cols are prediction vs target
- 
 
`names.py`
- replaces the `variable_names.yml` file
- import var names, constants, ect. from here in submodules (as an example, see one step diags workflow)

 
`__main__.py`
- load test data
- load model
- model.predict(test_data)
- standardize_data(test_data, predict_data)
   - dataset for each var that has >1 source, with coords for source
- create_diagnostics(high res, test, predict datasets)
 
`create_metrics.py`
- metrics needs:
   - net precipitation (total, ML, physics)
   - net heating (total, ML, physics)
   - qQ , dQ, Q
   - following do not need data source coord
       - land_sea_mask
       - delp
       - grid vars [lat, lon, latb, lonb, area]
 
  
`create_diagnostics.py`
- diagnostics needs:
 
   - T, sphum, sfc_temp: these are only used in the lower tropospheric stability calculation
   - grid
   - land_sea_mask
   - local time
   - 
- `
 
`plotting.py`: No calculations, just save plots and return the dict used by the report {section: [figures]}
- instead of hard coding in every variable/mask combo in the main plot all function, use config a la one step diags workflow that 
specifies all the usages of a given function
- plots
    - lower tropospheric stability
    - diurnal cycle of net heating, net precip, (can add just precip)
        - global/land/sea avg, or at specific location
    - map of some quantity [net precip, heating]
        - time avg
        - few snapshots
    - vertical profiles of dQ1/2 over land/ocean, drying/moistening columns


    
`data.py`
- `load_test_data()`
   - should work with future data loader design
 

- `standardize_data()`
   - not sure if right name- but the job of this is to get high res / target / prediction data into the format that lets subsequent functions do their thing.
   - e.g. same names across datasets for surface fluxes, radiation, etc.
 
 
`calc.py`: calculations for metrics, diagnostics
- LTS and related functions
- 
      
  
 
 
 
 
 
 
 
 









 
 
 
 
 
 
 








