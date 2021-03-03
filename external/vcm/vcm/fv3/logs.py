#!/usr/bin/env python
import re
from typing import Mapping, List, Tuple
from collections import defaultdict
from datetime import datetime
from toolz import curry
import dataclasses


__all__ = ["loads"]


def parse_date_stats_block(line):
    tokens = line.split()
    args = [int(tok) for tok in tokens]
    return datetime(*args)


@curry
def parse_pattern(pattern, line):
    water_vapor_regex = re.compile(pattern + r" *= (.*)")
    match = water_vapor_regex.search(line)
    if match:
        return float(match.group(1).strip())
    else:
        return None


def set_initial_date(dates):
    if dates[0] is None:
        initial_date = dates[1] - (dates[2] - dates[1])
        return [initial_date] + dates[1:]
    else:
        return dates


@dataclasses.dataclass
class FV3Log:
    dates: List[datetime]
    totals: Mapping[str, List[float]]
    ranges: Mapping[str, List[Tuple[float, float]]]


def loads(log: str) -> FV3Log:
    """Parse the statistics information from the logs for an FV3 simulation

    Does not parse python-related diagnostics. It extracts the statistical
    information from blocks with the following format::

                2016           8           1           1           0           0
        ZS      6849.180      -412.0000       231.8707
        PS max =    1052.038      min =    439.9279
        Mean specific humidity (mg/kg) above 75 mb=   3.993258
        Total surface pressure (mb) =    985.9462
        mean dry surface pressure =    983.2382
        Total Water Vapor (kg/m**2) =   27.51812
        --- Micro Phys water substances (kg/m**2) --
        Total cloud water=  3.4418099E-0
        Total rain  water=  8.9859152E-0
        Total cloud ice  =  3.6711354E-0
        Total snow       =  1.4170707E-0
        Total graupel    =  1.3332913E-0
        --------------------------------------------
        TE ( Joule/m^2 * E9) =   2.633417
        UA_top max =    138.3022      min =   -45.65562
        UA max =    138.3022      min =   -52.00678
        VA max =    60.31665      min =   -67.41895
        W  max =    36.12219      min =   -11.35321
        Bottom w max =    5.591572      min =   -6.320749
        Bottom: w/dz max =   0.2113777      min =  -0.2430508
        DZ (m) max =   -21.97151      min =   -5737.403
        Bottom DZ (m) max =   -21.97151      min =   -32.53793
        TA max =    319.3932      min =    171.7615
        OM max =    79.11412      min =   -91.14186
        ZTOP      40.79998       34.23963       39.11939
        SLP max =    1064.040      min =    953.2686
        ATL SLP max =    1022.554      min =    1010.739
        fv_GFS Z500   5700.141       5790.903       5459.689       5864.878
        Cloud_top_P (mb) max =   1.0000000E+10  min =    14.69188
        Surf_wind_speed max =    33.00807      min =   4.1885415E-0
        T_bot:      319.3932       216.0579       289.6649
        ZTOP max =    40.79998      min =    34.23963
        W5km max =    13.85566      min =   -5.952400
        max reflectivity =    64.90421      dBZ
        sphum max =   2.3002494E-02  min =   9.9999342E-10
        liq_wat max =   2.2458597E-03  min =  -1.3747283E-19
        rainwat max =   8.2634874E-03  min =  -1.4335402E-18
        ice_wat max =   5.2885562E-03  min =  -5.2392893E-20
        snowwat max =   8.4350556E-03  min =  -9.2761843E-19
        graupel max =   1.3989445E-02  min =  -4.0843832E-05
        o3mr max =   1.7852419E-05  min =   2.2205915E-09
        sgs_tke max =    130.6761      min =   9.8705533E-10

    Currently it only supports the min/max and total statistics.

    Returns:
        fv3log: an object with the parsed statistical information
        
    
    """
    lines = log.splitlines()
    min_max_regex = re.compile(r"(.*)max *= *([0-9E\-\.]+) *min = *([0-9E\-\+\.]+)")
    zs_regexp = re.compile(r"ZS *([0-9E\-\.]+) +([0-9E\-\.]+) +([0-9E\-\+\.]+)")
    totals = defaultdict(list)
    min_max = defaultdict(list)
    dates = []

    mean_patterns = {
        "total surface pressure": parse_pattern(r"Total surface pressure \(mb\)"),
        "total water vapor": parse_pattern(r"Total Water Vapor \(kg/m\*\*2\)"),
        "mean dry surface pressure": parse_pattern("mean dry surface pressure"),
        "total cloud water": parse_pattern("Total cloud water"),
        "total rain water": parse_pattern(r"Total rain\s+water"),
        "total cloud ice": parse_pattern("Total cloud ice"),
        "total snow": parse_pattern("Total snow"),
        "total graupel": parse_pattern("Total graupel"),
    }

    date = None
    for line in lines:

        if "fv_restart_end" in line:
            # the logs end with a block like this::
            #
            # fv_restart_end u    =    5404162460055465614
            # ...
            #  ZS   6849.180      -412.0000       231.8706
            #
            # This break avoids appending the final date twice.
            break

        try:
            date = parse_date_stats_block(line)
        except (TypeError, ValueError):
            pass

        match = zs_regexp.search(line)
        if match:
            # The statistics block begins with the date followed by ZS::
            #         2016           8           2           8           0           0
            #  ZS      6849.180      -412.0000       231.8707
            dates.append(date)

        match = min_max_regex.search(line)
        if match:
            name, max_, min_ = match.groups()
            name = name.strip()
            min_max[name].append((float(min_), float(max_)))

        for variable, parser in mean_patterns.items():
            val = parser(line)
            if val is not None:
                totals[variable].append(val)

    return FV3Log(set_initial_date(dates), dict(totals), dict(min_max))
