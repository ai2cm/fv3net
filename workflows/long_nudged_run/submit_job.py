import copy
import logging
import os
import uuid
from datetime import datetime, timedelta
import gcsfs
import yaml
import fv3config

logger = logging.getLogger("run_jobs")

START_DATE = datetime(2016, 1, 1, 0, 0, 0)
RUN_DURATION = timedelta(days=366)

RUN_NAME = f"nudged-2016.{uuid.uuid4()}"
BUCKET = "gs://vcm-ml-data/"
NUDGE_BUCKET = BUCKET + "2019-12-02-year-2016-T85-nudging-data"
IC_BUCKET = BUCKET + "2019-12-03-C48-20160101.00Z_IC"
OUTPUT_BUCKET = BUCKET + f"2019-12-12-baseline-FV3GFS-runs/nudged/C48/{RUN_NAME}/output"
CONFIG_BUCKET = BUCKET + f"2019-12-12-baseline-FV3GFS-runs/nudged/C48/{RUN_NAME}/config"
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"
LOCAL_RUNFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runfile.py")
LOCAL_DIAG_TABLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diag_table_with_nudge_dt"
)

LOCAL_NUDGE_NAMELIST = "nudge.nml"
NUDGE_FILENAME_PATTERN = "%Y%m%d_%HZ_T85LR.nc"
NUDGE_INTERVAL = 6  # hours

HOURS_IN_DAY = 24

nudging_file_list_location = os.path.join(CONFIG_BUCKET, "nudging_file_list")


def nudge_filename(date: datetime) -> str:
    return date.strftime(NUDGE_FILENAME_PATTERN)


def date_to_list(date: datetime) -> list:
    return [date.year, date.month, date.day, date.hour, date.minute, date.second]


def get_nudge_file_list(start_date: datetime, run_duration: timedelta) -> list:
    run_duration_hours = run_duration.days * HOURS_IN_DAY
    nudging_hours = range(0, run_duration_hours + NUDGE_INTERVAL, NUDGE_INTERVAL)
    time_list = [start_date + timedelta(hours=hour) for hour in nudging_hours]
    return [nudge_filename(time) for time in time_list]


def get_config(start_date: datetime, run_duration: timedelta) -> dict:
    config = fv3config.get_default_config()
    config["experiment_name"] = RUN_NAME
    config["diag_table"] = os.path.join(CONFIG_BUCKET, "diag_table")
    config["initial_conditions"] = IC_BUCKET
    config = fv3config.set_run_duration(config, RUN_DURATION)
    # make physics diagnostic output hourly
    config["namelist"]["atmos_model_nml"]["fhout"] = 1.0
    config["namelist"]["coupler_nml"].update(
        {"current_date": date_to_list(START_DATE), "dt_atmos": 900, "dt_ocean": 900}
    )
    config["namelist"]["fv_core_nml"].update({"nudge": True, "layout": [2, 2]})
    config["namelist"].update(fv3config.config_from_namelist(LOCAL_NUDGE_NAMELIST))
    patch_files = [
        fv3config.get_asset_dict(NUDGE_BUCKET, file, target_location="INPUT")
        for file in get_nudge_file_list(start_date, run_duration)
    ]
    patch_files.append(fv3config.get_asset_dict(CONFIG_BUCKET, "nudging_file_list"))
    config["patch_files"] = patch_files
    return config


def submit_job() -> None:
    fs = gcsfs.GCSFileSystem(project="VCM-ML")
    config = get_config(START_DATE, RUN_DURATION)
    job_name = config["experiment_name"]
    config_location = os.path.join(CONFIG_BUCKET, "fv3config.yml")
    with fs.open(config_location, "w") as config_file:
        config_file.write(yaml.dump(config))
    with fs.open(nudging_file_list_location, "w") as nudging_filelist:
        nudging_filelist.write("\n".join(get_nudge_file_list(START_DATE, RUN_DURATION)))
    runfile_location = os.path.join(CONFIG_BUCKET, "runfile.py")
    fs.put(LOCAL_RUNFILE, runfile_location)
    fs.put(LOCAL_DIAG_TABLE, os.path.join(CONFIG_BUCKET, "diag_table"))
    fv3config.run_kubernetes(
        config_location,
        OUTPUT_BUCKET,
        DOCKER_IMAGE,
        runfile=runfile_location,
        jobname=job_name,
        namespace="default",
        memory_gb=14,
        cpu_count=24,
        gcp_secret="gcp-key",
        image_pull_policy="Always",
    )
    logger.info(f"Submitted {job_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    submit_job()
