"""
Functions that take in datasets and save all plots by
iterating over all variables in a diagnostic/plot type in the config.
They return a dict of {html section: [figures in section]}

variables in datasets should already be ready to plot along coords
"""

def plot_diagnostics(ds: xr.Dataset, plot_config, output_dir) -> Dict[str, List[str]]:
    pass


def plot_metrics(ds: xr.Dataset, plot_config, output_dir) -> Dict[str, List[str]]:
    pass


def plot_lts(ds: xr.Dataset, output_dir) -> Dict[str, List[str]]:
    # lower tropospheric stability plot is its own thing
    pass

