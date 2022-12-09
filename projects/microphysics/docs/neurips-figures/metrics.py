import pathlib
import wandb
from fv3net.diagnostics.prognostic_run.emulation import single_run
import vcm


api = wandb.Api()


def _get_runs(group):
    runs = api.runs(
        "ai2cm/microphysics-emulation",
        filters={"group": {"$regex": group}, "jobType": "prognostic_run"},
    )
    for run in runs:
        yield run.group, run.config["rundir"]


groups = pathlib.Path("final_groups.txt").read_text().splitlines()
print("group", "global", "lat < -75", "Lat > -75", sep=",")
for group in groups:
    for group, rundir in _get_runs(group):
        try:
            ds = single_run.open_rundir(rundir)
        except Exception:
            continue

        avgs = []
        for w in [ds.area, ds.area.where(ds.lat < -75), ds.area.where(ds.lat > -75)]:
            ans = ds.cloud_water_mixing_ratio.sel(time="2016-07-15T00:00:00")
            ans = vcm.mass_integrate(
                ans, ds.pressure_thickness_of_atmospheric_layer, "z"
            )
            ans = vcm.weighted_average(ans, w)
            ans = ans.load().item()
            avgs.append(ans)

        print(group, *avgs, sep=",")
