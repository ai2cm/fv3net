IMAGE = us.gcr.io/vcm-ml/prognostic_run:mean_nudging
FV3CONFIG = nudge_mean_$*.yml
RUNFILE = mean_nudging_runfile.py
GCS_OUTDIR = gs://vcm-ml-data/2020-03-30-learned-nudging-FV3GFS-runs/$*


build:
	docker build . -f Dockerfile -t $(IMAGE)

push:
	docker push $(IMAGE)

run_remote: run_remote_T run_remote_T_ps run_remote_T_ps_u_v

run_remote_%:
	python submit_job.py --run-yaml $(FV3CONFIG) --outdir $(GCS_OUTDIR)

run_local: run_local_T run_local_T_ps run_local_T_ps_u_v

run_local_%:
	fv3run --dockerimage $(IMAGE) --runfile $(RUNFILE) $(FV3CONFIG) rundir_$*

clean:
	rm -rf rundir_*

.PHONY: build push run_local clean
