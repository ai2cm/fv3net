### Purpose
Create diagnostic plots for one-timestep or multi-timestep runs.

#### Overview
Creation of diagnostic plots is handled through a plotting config YAML file. Each entry in the config 
corresponds to one figure and contains the information needed to calculate the diagnostic quantity 
and construct the figure.
- Selection on dimension indices via `isel()` occurs first. The index or slice to select on is given in the 
`dim_slices`. If a single value is provided for a dimension, the data will be selected on `isel(dim=index)`. If more
than one value is provided, the data will be selected on `isel(dim=slice(index0, index1)`.  
- Functions to perform calculations, transformations, etc. on the data are specified **in order of execution** along
with their keyword arguments. These functions used to compute the diagnostics are kept in `vcm.calc.diag_ufuncs`. 
The functions applied should result in the creation of the named `diagnostic_variable`s if they do not already exist
in the dataset.
- The plot is generated for the `diagnostic_variable` in the config. This can be provided as a list if multiple
quantities can be plotted in the same figure (i.e. not a map). The axes are determined by the specified 
`plotting_function`.

Below is an example of how to specify a diagnostic in the config yaml file. More examples are provided at the bottom
of this readme doc.
Keep the dash at the top of each plot entry as it denotes that each plot config block 
is an element in a list.

    -
      plot_name: dT_dt at 825 HPa  # this will be the header for the plot in the html report
      plotting_function: plot_time_series # function in fv3net.diagnostics.visualize that is used to create plot
      diagnostic_variable: # can provide single string or a list of variables to plot simultaneously
          - dT_dt
          - dT_dt_C48
      dim_slices:  # args for use in subselected data via data.isel()
        forecast_time:
          - 0
        initialization_time:
          - null
          - null
        grid_xt: 
          - 20
        grid_yt:
          - 5
        tile:
          - 3
        pfull:
          - 60
      function_specifications: # function names from vcm.calc.diag_ufuncs, followed by the function args
        - remove_extra_dim:
      plot_params: # arguments to pass to plotting function
        xlabel: time
        ylabel: dT/dt
        
        
#### Adding a new diagnostic
1. If needed, add user defined function(s) from `vcm.calc.diag_ufuncs` that computes the desired quantity and 
adds it as a variable in the dataset. The functions should be provided in order of piping. 
2. Create an entry for the diagnostic in the config yaml file. See instructions above regarding the config format.
3. If the current plotting functions in `fv3net.diagnostics.visualize` do not suffice for your use case, you can add 
another plot function to that module. Use the function name as the config entry `plotting_function`. Unit tests for
new plot functions need to be run once locally to generate a baseline plot future comparisons, which should
be stored in `tests/baseline_plots`.



### Example: automatically create all plots from config yaml
```
python -m fv3net.diagnostics.create_all_diagnostics.py \
 --config-file default_config.yaml \
 --output-dir diag_output \
 --gcs-run-dir gs://bucket/data-location
```


### Example: interactive usage
The diagnostic suite is designed for automated usage, but if you want to use it to recreate a 
figure in interactive mode, you can create the `PlotConfig` python object yourself. Note that
if you have multiple user defined functions being applied, you'll need to make sure their order 
and the order of plot kwargs matches what you want. 
```
from fv3net.diagnostics.utils import PlotConfig
from fv3net.diagnostics.visualize import create_plot

# import the user defined funcs that are needed from vcm
from vcm.calc.diag_ufuncs import mean_over_dim

...
# assuming that the dataset ds is already loaded

plot_config = PlotConfig(
    diagnostic_variable="mean_diag_var",
    plot_name="test time series sliced",
    plotting_function="plot_time_series",
    dim_slices={"initialization_time": slice(None, 50, None)},
    functions=[mean_over_dim],
    function_kwargs=[
        {"dim": "pfull", "var_to_avg": "diag_var", "new_var": "mean_diag_var"}
    ],
    plot_params={"xlabel": "time [d]", "ylabel": "mean diag var"},
)
fig = create_plot(ds, plot_config)
```


### Additional examples of config entries for plots
    # example of faceted map plot
    -
      plot_name: dT_dt, vertical int
      plotting_function: plot_diag_var_map
      diagnostic_variable: 
          - dT_dt_vertical_int
      dim_slices:
        forecast_time:
          - 0
        initialization_time:
          - 25
          - 35
      function_specifications:  
        - remove_extra_dim:
        - apply_weighting:
            var_to_weight: dT_dt
            weighting_var: delp
            weighting_dims: pfull
        - sum_over_dim:
             dim: pfull
             var_to_sum: dT_dt
             new_var: dT_dt_vertical_int
      plot_params:
        col: initialization_time  
        vmin: -0.0001
        vmax: 0.0001
        cbar_label: dT_dt vertical integral
        col_wrap: 5
        
    # example of single map plot
    -
      plot_name: dT_dt, vertical int
      plotting_function: plot_diag_var_map
      diagnostic_variable: dT_dt_vertical_int
      dim_slices:
        forecast_time:
          - 0
        initialization_time:
          - 27
      function_specifications:
        - remove_extra_dim:
        - apply_weighting:
            var_to_weight: dT_dt
            weighting_var: delp
            weighting_dims: pfull
        - sum_over_dim:
             dim: pfull
             var_to_sum: dT_dt
             new_var: dT_dt_vertical_int
      plot_params:
        vmin: -0.0001
        vmax: 0.0001
