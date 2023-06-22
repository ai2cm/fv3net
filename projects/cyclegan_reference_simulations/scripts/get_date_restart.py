import argparse
import os
import re

import cftime
import xarray as xr
import yaml

import fv3config


UNITS = {"seconds", "minutes", "hours", "days", "months"}
NON_MONTH_UNITS = {"seconds", "minutes", "hours", "days"}

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("restart_directory")
parser.add_argument("segment")
parser.add_argument("--end-date", action="store_true")
args, extra_args = parser.parse_known_args()


def get_duration_config(config):
    coupler_nml = config["namelist"].get("coupler_nml", {})
    return {unit: coupler_nml.get(unit, 0) for unit in UNITS}


def duration_timedelta_representable(config):
    duration_config = get_duration_config(config)
    return duration_config["months"] == 0


def duration_months_representable(config):
    duration_config = get_duration_config(config)
    return all(duration_config[unit] == 0 for unit in NON_MONTH_UNITS)


def parse_date_from_line(line, coupler_res_filename):
    """Parses a date from a line in a coupler.res file. Adapted from fv3config."""
    date = [int(d) for d in re.findall(r"\d+", line)]
    if len(date) != 6:
        raise ValueError(
            f"{coupler_res_filename} does not have a valid date in the given line "
            f"(line must contain six integers)"
        )
    return date


def parse_date_from_restart_directory(restart_directory):
    """Adapted from fv3config."""
    coupler_res_filename = os.path.join(restart_directory, "coupler.res")
    with open(coupler_res_filename, mode="r") as f:
        lines = f.readlines()
        current_date = parse_date_from_line(lines[2], coupler_res_filename)
    return cftime.DatetimeJulian(*current_date)


with open(args.config, "r") as file:
    config = yaml.safe_load(file)

initial_date = parse_date_from_restart_directory(args.restart_directory)

if duration_timedelta_representable(config):
    duration = fv3config.get_run_duration(config)
    segment = int(args.segment)
    if args.end_date:
        segment = segment + 1

    date = initial_date + segment * duration
    print(date.strftime("%Y%m%d%H"))
elif duration_months_representable(config) and initial_date.day == 1:
    duration = get_duration_config(config)["months"]
    segment = int(args.segment)
    if args.end_date:
        segment = segment + 1
    freq = f"{duration}MS"
    periods = segment + 1  # segment == 1 to return initial date
    date_range = xr.cftime_range(initial_date, freq=freq, periods=periods)
    date = date_range.values[-1]
    print(date.strftime("%Y%m%d%H"))
else:
    raise ValueError("Segment length and initial date combination not supported.")
