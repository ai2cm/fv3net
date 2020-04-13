IMAGE = us.gcr.io/vcm-ml/learned_nudging_run:v0.1.1
RUNFILE = mean_nudging_runfile.py
TEMPLATE = fv3config_template.yml
LOCAL_OUTDIR = rundir/nudge_mean_$*
LOCAL_CONFIGDIR = $(LOCAL_OUTDIR)/config
LOCAL_FV3CONFIG = $(LOCAL_CONFIGDIR)/fv3config.yml
GCS_OUTDIR = gs://vcm-ml-data/2020-03-30-learned-nudging-FV3GFS-runs/nudge_mean_$*
GCS_CONFIGDIR = $(GCS_OUTDIR)/config
GCS_FV3CONFIG = $(GCS_CONFIGDIR)/fv3config.yml

run_all_remote: run_remote_T run_remote_T_ps run_remote_T_ps_u_v

run_remote_%: prepare_remote_%
	python submit_job.py --dockerimage $(IMAGE) --runfile $(RUNFILE) $(GCS_FV3CONFIG) $(GCS_OUTDIR)

run_local_%: prepare_local_%
	fv3run --dockerimage $(IMAGE) --runfile $(RUNFILE) $(LOCAL_FV3CONFIG) $(LOCAL_OUTDIR)

prepare_remote_%:
	python prepare_config.py $(TEMPLATE) $* $(GCS_CONFIGDIR)

prepare_local_%:
	python prepare_config.py $(TEMPLATE) $* $(LOCAL_CONFIGDIR)

clean:
	rm -rf rundir configdir

.PHONY: run_all_remote
