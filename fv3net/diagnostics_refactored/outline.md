### Refactored offline diagnostics workflow
Other concerns
- do we want to continue to create figures like the "diurnal cycle across workflow", or was that a one off for the presentation
- OOM issues when loading data
   - `.load()` only necessary values
   - make sure the grid isn't getting a time dim
- is there a significant cost increase for copying data to the local drive, then reading from there, then delete after use? this is much faster that loading from GCS and we don't have to worry about retrying upon read errors
- just hardcode the x/y/tile coords for the particular location points in the constants rather than doing the lat/lon lookup each time this runs.
- naming convention when refering to the data source: pred/target/hires
 
 
diagnostics
- time avg col integrated map plot of dQ1/2, Q1/2, precip, where cols are prediction vs target
- snapshot map plots for a couple timesteps of dQ1, dQ2, Q1, 
- vertical profiled dQ1/2 for pos/neg heating columns
- diurnal cycles of P-E, net heating over land/sea/global


metrics
- RMSE, bias for column integrated Q2, Q1
- mass weighted RMSE for dQ1, dQ2


`config_diagnostics.yaml`
- contains list of diagnostics and variables that each diagnostic is calculated for
- used by diagnostics and plotting to iterate over datasets/plots to create


`names.py`
- replaces the `variable_names.yml` file
- import var names, constants, ect. from here in submodules (as an example, see one step diags workflow)

 
`__main__.py`
- load test data
- load model
- model.predict(test_data)
- data_arrays: create dict to organize all the data arrays for quantities used in metrics/diagnostics 
   - referenced by keys "{var}_{source}" e.g. "dQ1_target"
   - if a new quantity is needed for diagnostics, e.g. local time, calculations occur at this step to insert 
   it as another data array 
   - this dict get passed to the the functions that create metrics/diagnostics
- create and save datasets for metrics and diagnostics
- save plots, plot functions take in datasets from previous step
- save report

 
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
   - dQ1/2, Q1/2

 
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
      
  
 
 
 
 
 
 
 
 









 
 
 
 
 
 
 








