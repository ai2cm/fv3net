FV3CONFIG = nudge_mean_$*.yml
RUNFILE = mean_nudging_runfile.py
GCS_OUTDIR = gs://vcm-ml-data/2020-03-30-learned-nudging-FV3GFS-runs/$*


run_all: run_T run_T_ps run_T_ps_u_v

run_%:
	python ../single_fv3gfs_run/submit_job.py --run-yaml $(FV3CONFIG) --bucket $(GCS_OUTDIR)

clean:
	rm -rf rundir_*

.PHONY: build push run_local clean
