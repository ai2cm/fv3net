from .data import (
    load_test_data, load_model, offline_predction, get_var_arrays)
from .create_metrics import create_metrics_dataset, calc_scalar_metrics
from .create_diagnostics import create_diagnostics_dataset
import report



# outline of main function
diagnostics_config = yaml.load(diag_config_path)
ds_test = load_test_data(test_data_path)
model = load_model(model_path)
ds_prediction = offline_predction(model, ds_test)

# ex. {dQ1_target: DataArray, dQ1_predict: DataArray, net_precip_total_hires: DataArray, net_precip_total_target, DataArray, ...}
data_arrays = get_var_arrays(ds_test, ds_prediction)

# any additional function calls for calculating arrays needed in later functions goes here,
# e.g. local time, total Q, etc.
data_arrays["local_time"] = local_time(...)
data_arrays["Q1_target"] = total_Q(ds_test, "Q1")
...

# save datasets and json of metrics for later plots
ds_metrics = create_metrics_dataset(data_arrays)
ds_metrics.to_netcdf(...)
scalar_metrics = calc_scalar_metrics(data_arrays)
json.dumps(scalar_metrics, ...)

ds_diagnostics = create_diagnostics_dataset(data_arrays, diagnostics_config)
ds_lts = create_lower_trop_stability_dataset(data_arrays)  # note in function def about why this is separate
ds_diagnostics.to_netcdf(...)
ds_lts.to_netcdf(...)


# plots
report_sections_diagnostics = plot_diagnostics(ds_diagnostics, diagnostics_config, output_dir)
report_sections_lts = plot_lts(ds_lts, output_dir)
report_sections_metrics = plot_metrics(ds_metrics, diagnostics_config, output_dir)
report_sections_timesteps = plot_timestep_counts(...)


# create report
_write_report(output_dir, sections, metadata, title)