import argparse

import cftime
import xarray as xr
import yaml

import fv3config


UNITS = {"seconds", "minutes", "hours", "days", "months"}
NON_MONTH_UNITS = {"seconds", "minutes", "hours", "days"}

parser = argparse.ArgumentParser()
parser.add_argument("config")
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


with open(args.config, "r") as file:
    config = yaml.safe_load(file)

initial_date_tuple = config["namelist"]["coupler_nml"]["current_date"]
initial_date = cftime.DatetimeJulian(*initial_date_tuple)

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
