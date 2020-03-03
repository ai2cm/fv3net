
import pandas as pd
from vcm.fv3_restarts import open_restarts, get_restart_times, _parse_time, _parse_time_string
from pathlib import Path
import os
import fsspec
import vcm.combining

from toolz import memoize


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


