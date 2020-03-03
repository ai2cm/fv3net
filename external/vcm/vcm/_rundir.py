
import pandas as pd
from vcm.fv3_restarts import get_restart_times, _parse_time, _parse_time_string
from pathlib import Path
import os
import fsspec
import vcm.combining

from toolz import memoize

from fv3_restarts import SCHEMA_CACHE, standardize_metadata


def split(url):
    path = Path(url)
    no_timestamp = path.name.endswith("INPUT") or path.name.endswith("RESTART")
    
    if no_timestamp:
        parent = path.parent
    else:
        parent = path.parent.parent

    return str(parent), str(path.relative_to(parent))

def append_if_not_present(list, item):
    if item not in list:
        list.append(item)


def get_prefixes(fs, url):
    prefixes = [str(Path(url) / 'INPUT')]
    restarts = fs.glob(url + '/RESTART/????????.??????.*')
    for restart in restarts:
        time = _parse_time(Path(restart).name)
        append_if_not_present(prefixes, str(Path(url) / 'RESTART' / time))
    prefixes.append(str(Path(url) / 'RESTART'))
    return prefixes


def file_prefix_to_times(fs, prefix):
    root, pref = split(prefix)
    mapping = get_prefix_time_mapping(fs, root)
    return _parse_time_string(Path(root).name), mapping[os.path.join(root, pref)]


def replace_file_prefix_coord(fs, ds):
    def _compute_times(prefix):
        prefix = prefix
        init, final = file_prefix_to_times(fs, prefix)
        return init, final - init

    times = [_compute_times(prefix.item()) for prefix in ds.file_prefix]
    names = ['init_time', 'lead_time']
    times = pd.MultiIndex.from_tuples(times, names=names)
    return  ds.assign_coords(file_prefix=times)#.unstack('file_prefix')
    return vcm.combining.move_dims_to_front(unstacked, names)


@memoize(key=lambda args, kwargs: args[1])
def get_prefix_time_mapping(fs, url):
    times = get_restart_times('gs://' + url)
    prefixes = get_prefixes(fs, url)
    return dict(zip(prefixes, times))


def sorted_file_prefixes(prefixes):

    return sorted(
            prefix
            for prefix in prefixes
            if prefix not in ["INPUT", "RESTART"]
    )

def _get_initial_time(prefix):
    return str(Path(prefix).parent.parent.name)


def get_restart_times(url: str) -> Sequence[cftime.DatetimeJulian]:
    """Reads the run directory's files to infer restart forecast times

    Due to the challenges of directly parsing the forecast times from the restart files,
    it is more robust to read the ime outputs from the namelist and coupler.res
    in the run directory. This function implements that ability.

    Args:
        url (str): a URL to the root directory of a run directory.
            Can be any type of protocol used by fsspec, such as google cloud storage
            'gs://path-to-rundir'. If no protocol prefix is used, then it will be
            assumed to be a path to a local file.

    Returns:
        time Sequence[cftime.DatetimeJulian]: a list of time coordinates
    """
    proto, namelist_path = _get_namelist_path(url)
    config = _config_from_fs_namelist(proto, namelist_path)
    initialization_time = _get_current_date(config, url)
    duration = _get_run_duration(config)
    interval = _get_restart_interval(config)
    forecast_time = _get_forecast_time_index(initialization_time, duration, interval)
    return forecast_time


def _split_url(url):

    try:
        protocol, path = url.split("://")
    except ValueError:
        protocol = "file"
        path = url

    return protocol, path


def _get_file_prefix(dirname, path):
    if dirname.endswith("INPUT"):
        return "INPUT/"
    elif dirname.endswith("RESTART"):
        try:
            return os.path.join("RESTART", parse_timestep_str_from_path(path))
        except ValueError:
            return "RESTART/"


def _sort_file_prefixes(ds, url):

    if "INPUT/" not in ds.file_prefix:
        raise ValueError(
            "Open restarts did not find the input set "
            f"of restart files for run directory {url}."
        )
    if "RESTART/" not in ds.file_prefix:
        raise ValueError(
            "Open restarts did not find the final set "
            f"of restart files for run directory {url}."
        )

    intermediate_prefixes = sorted(
        [
            prefix.item()
            for prefix in ds.file_prefix
            if prefix.item() not in ["INPUT/", "RESTART/"]
        ]
    )

    return xr.concat(
        [
            ds.sel(file_prefix="INPUT/"),
            ds.sel(file_prefix=intermediate_prefixes),
            ds.sel(file_prefix="RESTART/"),
        ],
        dim="file_prefix",
    )


def _parse_category(path):
    cats_in_path = {category for category in RESTART_CATEGORIES if category in path}
    if len(cats_in_path) == 1:
        return cats_in_path.pop()
    else:
        # Check that the file only matches one restart category for safety
        # it not clear if this is completely necessary, but it ensures the output of
        # this routine is more predictable
        raise ValueError("Multiple categories present in filename.")


def _get_tile(path):
    """Get tile number

    Following python, but unlike FV3, the first tile number is 0. In other words, the
    tile number of `.tile1.nc` is 0.

    This avoids confusion when using the outputs of :ref:`open_restarts`.
    """
    tile = re.search(r"tile(\d)\.nc", path).group(1)
    return int(tile) - 1


def _is_restart_file(path):
    return any(category in path for category in RESTART_CATEGORIES) and "tile" in path


def _restart_files_at_url(url):
    """List restart files with a given initial and end time within a particular URL

    Yields:
        (time, restart_category, tile, protocol, path)

    """
    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if _is_restart_file(file):
                file_prefix = _get_file_prefix(root, file)
                tile = _get_tile(file)
                category = _parse_category(file)
                yield file_prefix, category, tile, proto, path


def _get_namelist_path(url):

    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            if _is_namelist_file(file):
                return proto, os.path.join(root, file)


def _is_namelist_file(file):
    return "input.nml" in file


def _get_coupler_res_path(url):

    proto, path = _split_url(url)
    fs = fsspec.filesystem(proto)

    for root, dirs, files in fs.walk(path):
        for file in files:
            if _is_coupler_res_file(root, file):
                return proto, os.path.join(root, file)


def _is_coupler_res_file(root, file):
    return "INPUT/coupler.res" in os.path.join(root, file)


def _config_from_fs_namelist(proto, namelist_path):
    fs = fsspec.filesystem(proto)
    with fs.open(namelist_path, "rt") as f:
        return _to_nested_dict(f90nml.read(f).items())


def _to_nested_dict(source):
    return_value = dict(source)
    for name, value in return_value.items():
        if isinstance(value, f90nml.Namelist):
            return_value[name] = _to_nested_dict(value)
    return return_value


def _get_current_date(config, url):
    """Return current_date as a datetime from configuration dictionary
    Note: Mostly copied from fv3config, but with fsspec capabilities added
    """
    force_date_from_namelist = config["coupler_nml"].get(
        "force_date_from_namelist", False
    )
    # following code replicates the logic that the fv3gfs model
    # uses to determine the current_date
    if force_date_from_namelist:
        current_date = config["coupler_nml"].get("current_date", [0, 0, 0, 0, 0, 0])
    else:
        try:
            proto, coupler_res_filename = _get_coupler_res_path(url)
            current_date = _get_current_date_from_coupler_res(
                proto, coupler_res_filename
            )
        except TypeError:
            current_date = config["coupler_nml"].get("current_date", [0, 0, 0, 0, 0, 0])
    return datetime(
        **{
            time_unit: value
            for time_unit, value in zip(
                ("year", "month", "day", "hour", "minute", "second"), current_date
            )
        }
    )


def _get_current_date_from_coupler_res(proto, coupler_res_filename):
    """Return a timedelta indicating the duration of the run.
    Note: Mostly copied from fv3config, but with fsspec capabilities added
    """
    fs = fsspec.filesystem(proto)
    with fs.open(coupler_res_filename, "rt") as f:
        third_line = f.readlines()[2]
        current_date = [int(d) for d in re.findall(r"\d+", third_line)]
        if len(current_date) != 6:
            raise ValueError(
                f"{coupler_res_filename} does not have a valid current model time"
                "(need six integers on third line)"
            )
    return current_date


def _get_run_duration(config):
    """Return a timedelta indicating the duration of the run.
    Note: Mostly copied from fv3config
    """
    coupler_nml = config.get("coupler_nml", {})
    months = coupler_nml.get("months", 0)
    if months != 0:  # months have no set duration and thus cannot be timedelta
        raise ValueError(f"namelist contains non-zero value {months} for months")
    return timedelta(
        **{
            name: coupler_nml.get(name, 0)
            for name in ("seconds", "minutes", "hours", "days")
        }
    )


def _get_restart_interval(config):
    config = config["coupler_nml"]
    return timedelta(
        seconds=(config.get("restart_secs", 0) + 86400 * config.get("restart_days", 0))
    )


def _get_forecast_time_index(initialization_time, duration, interval):
    """Return a list of cftime.DatetimeJulian objects for the restart output
    """
    if interval == timedelta(seconds=0):
        interval = duration
    end_time = initialization_time + duration
    return [
        cftime.DatetimeJulian(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
        )
        for timestamp in pd.date_range(
            start=initialization_time, end=end_time, freq=interval
        )
    ]