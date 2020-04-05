from scipy.stats import binned_statistic_2d
import xarray as xr
from vcm.calc.calc import local_time
from vcm.visualize import plot_cube, mappable_var
from vcm.visualize.plot_diagnostics import plot_diurnal_cycle
from fv3net.diagnostics.data_funcs import (
    get_latlon_grid_coords_set,
    EXAMPLE_CLIMATE_LATLON_COORDS,
)
from typing import Mapping

def make_all_plots(states_and_tendencies: xr.Dataset, output_dir: str) -> Mapping:
    """ Makes figures for predictions on test data

    Args:
        states_and_tendencies: processed dataset of outputs from one-step
            jobs, containing states and tendencies of both the hi-res and
            coarse model, averaged across initial times for various 2-D,
            variables, global/land/sea mean time series, and global/land/sea
            mean time-height series, i=output from
            fv3net.diagnostics.one_step_jobs
        output_dir: location to write figures to

    Returns:
        dict of header keys and image path list values for passing to the html
        report template
    """
    
    report_sections = {}
    
    figs = map_plot_ml_frac_of_total(ds)
    fig_pe_ml, fig_pe_ml_frac, fig_heating_ml, fig_heating_ml_frac = figs
    fig_pe_ml.savefig(os.path.join(output_dir, "dQ2_vertical_integral_map.png"))
    fig_pe_ml_frac.savefig(os.path.join(output_dir, "dQ2_frac_of_PE.png"))
    fig_heating_ml.savefig(os.path.join(output_dir, "dQ1_vertical_integral_map.png"))
    fig_heating_ml_frac.savefig(os.path.join(output_dir, "dQ1_frac_of_heating.png"))
    report_sections["ML model contributions to Q1 and Q2"] = [
        "dQ2_vertical_integral_map.png",
        "dQ2_frac_of_PE.png",
        "dQ1_vertical_integral_map.png",
        "dQ1_frac_of_heating.png",
    ]