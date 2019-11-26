### Creating diagnostic plots 
Create diagnostic plots for one-timestep or multi-timestep runs.

#### Overview
Creation of diagnostic plots is handled through a plotting config YAML file. Each entry in the config 
specifies how each diagnostic is calculated. 
- Selection on dimension indices via `isel()` occurs first. The index or slice to select on is given in the 
`dim_slices`. If a single value is provided for a dimension, the data will be selected on `isel(dim=index)`. If more
than one value is provided, the data will be selected on `isel(dim=slice(index0, index1)`.  
- Functions to perform calculations, transformations, etc. on the data are specified **in order of execution** along
with their keyword arguments. These functions used to compute the diagnostics are kept in `vcm.diagnostic.ufuncs`. 
The functions applied should result in the creation of the named `diagnostic_variable` if it does not already exist
in the dataset.
- The plot is generated for the `diagnostic_variable` in the config. The axes are determined by the specified 
`plot_type`.

Below is an example of how to specify a diagnostic in the config YAML file. Keep the dash at the top as it denotes
that each plot config block is an entry in a list.

    -
        plot_name: Precipitable water
        plot_type: time_series  # see plot.PLOT_TYPES for the list of options
        diagnostic_variable: PW  # diagnostic variable. can either exist in raw data or be calculated by user defined functions
        dim_slices:  # subselect data on ds.isel(dim=index) or ds.isel(dim=slice)
        initialization_time:
          - null
          - 20
        pfull:
          - null
          - null
        function_specifications:  # MUST PROVIDE IN ORDER OF EXECUTION
          - sum:    # function name that corresponds to key in FUNCTION_MAP
              dim_to_sum: pfull   # function kwargs
              var_to_sum: sphum
              new_var: PW


#### Adding a new diagnostic
1. If needed, add a user defined function in `vcm.diagnostic.ufuncs` that computes the desired quantity and 
adds it as a variable in the dataset.
2. Create an entry for the diagnostic in the config yaml file. See instructions above regarding the config format.

Current plot options (will add as more functions added):
- map plot a single variable for a single snapshot or time average
- time series
- zonal avg



### Example: automatically create all plots from config yaml
```
python run_all.py --config-file default_config.yaml --output-dir diag_output --gcs-run-dir gs://bucket/data-location
```

### Example: create individual plots in interactive mode
Individual diagnostics can be created interactively by either i) making the plot config yaml for the desired plots 
and reading it in the notebook or ii) filling in the `PlotConfig` in a notebook.

option (i): Safer way of defining the diagnostics to plot since it is clear in the yaml which plot kwargs go 
with each function.
```
from vcm.diagnostic.plot import create_plot
from vcm.diagnostic.utils import load_config

ds = xr.open_dataset( ... )

plot_configs = load_configs('test_config.yaml')
for plot_config in plot_configs:
    create_plot(ds, plot_config)
```

option (ii): Not recommended, but might be faster in the case where you're testing out a new function and have it
defined outside of `vcm.diagnostic.ufuncs`. 
You can skip writing a config file if you create a `PlotConfig` for each diagnostic yourself.
In this case, make sure that the functions are given in order that they should be applied, and the kwargs 
dicts are also in the correct order of function application.
```
from vcm.diagnostic.utils import PlotConfig
from vcm.diagnostic.plot import create_plot

from vcm.diagnostic import ufuncs 

ds = xr.open_dataset( ... )

plot_config = PlotConfig(
    var='precip',
    plot_name='avg_precip',
    plot_type='time_series',
    dim_slices=[{'initialization_time': slice(None, 25), 'lat': slice(20, 25)}],
    functions=[ufuncs.mean_over_dim, ],
    function_kwargs=[{'dim': 'lon'}]
)
create_plot(ds, plot_config)
```