### Creating diagnostic plots 
Create diagnostic plots for one-timestep or multi-timestep runs.

#### Overview
Creation of diagnostic plots is handled through a plotting config YAML file. Each entry in the config 
specifies how each diagnostic is calculated. 
- Selection on dimension indices via `isel()` occurs first. The index or slice to select on is given in the 
`dim_slices`. If a single value is provided for a dimension, the data will be selected on `isel(dim=index)`. If more
than one value is provided, the data will be selected on `isel(dim=slice(index0, index1)`.  
- Functions to perform calculations, transformations, etc. on the data are specified **in order of execution** along
with their keyword arguments. These functions used to compute the diagnostics are kept in `vcm.calc.diag_ufuncs`. 
The functions applied should result in the creation of the named `diagnostic_variable` if it does not already exist
in the dataset.
- The plot is generated for the `diagnostic_variable` in the config. The axes are determined by the specified 
`plotting_function`.

Below is an example of how to specify a diagnostic in the config YAML file. Keep the dash at the top as it denotes
that each plot config block is an entry in a list.

    -
        plot_name: Precipitable water
        plotting_function: time_series  # must exit in 
        
        diagnostic_variable: PW  # diagnostic variable. can either exist in raw data or be calculated by user defined functions
        dim_slices:  # subselect data on ds.isel(dim=index) or ds.isel(dim=slice)
        initialization_time:  
          - null
          - 100
          - 5
        pfull:
          - null
          - null
        function_specifications:  # MUST PROVIDE IN ORDER OF EXECUTION
          - sum_over_dim:    # function name that exists in vcm.calc.diag_ufuncs
              dim_to_sum: pfull   # function kwargs
              var_to_sum: sphum
              new_var: PW
              apply_delp_weighting: True
        plot_kwargs:
            xlabel: time
            ylabel: PW [kg/kg]

#### Adding a new diagnostic
1. If needed, add a user defined function in `vcm.calc.diag_ufuncs` that computes the desired quantity and 
adds it as a variable in the dataset.
2. Create an entry for the diagnostic in the config yaml file. See instructions above regarding the config format.
3. If the current plotting functions in `fv3net.diagnostics.visualize` do not suffice for your use case, you can add 
another plot function to that module. Use the function name as the config entry `plotting_function`.


Current plot options


### Example: automatically create all plots from config yaml
```
python create_all_diagnostics.py --config-file default_config.yaml --output-dir diag_output --gcs-run-dir gs://bucket/data-location
```
