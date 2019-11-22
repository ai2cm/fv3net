from dataclasses import dataclass
import ufuncs

# TODO: fix the pressure levels to actually reflect the labels

ONE_STEP_DIAGNOSTICS = {
    'Q1_vertical_avg':
        DiagSpecifier(
            var='Q1',
            vertical_slice=slice(None, None),
            functions={ufuncs.mean: {'dim': 'pfull'}) ,
    'Q1_lower_troposphere_avg':
        DiagSpecifier('Q1', vertical_slice=slice(30, 40)),
    'Q1_upper_troposphere_avg':
        DiagSpecifier('Q1', vertical_slice=slice(20, 30)),
    'Q2_vertical_avg':
        DiagSpecifier('Q2', vertical_slice=slice(None, None)),
    'Q2_lower_troposphere_avg':
        DiagSpecifier('Q2', vertical_slice=slice(30, 40)),
    'Q2_upper_troposphere_avg':
        DiagSpecifier('Q2', vertical_slice=slice(20, 30)) }



MULTI_STEP_DIAGNOSTICS = [
    'precipitable_water_vertical_avg_time_series',
    'precipitation_time_series',
    'abs_surface_pressure_tendency_time_series',
    'precipitation_usa_land_time_series',
    'upward_long_wave_flux_diff_ceres_time_avg',
    'upward_short_wave_flux_diff_ceres_time_avg',
    'precipitable_water_time_avg'
]


@dataclass
class DiagSpecifier:
    """
    Used to specify data array to plot.
    e.g.
        ds[var] \
            .isel(pfull=vertical_slice, time_slice=time_slice) \
            .pipe(functions.key,  **functions.value)
    """
    var: str
    vertical_slice: slice
    time_slice: slice = None
    functions: dict = None  # {function to pipe: kwargs}



