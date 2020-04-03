IMAGE = us.gcr.io/vcm-ml/prognostic_run:mean_nudging
CONFIGDIR = configdir/nudge_mean_$*
FV3CONFIG = $(CONFIGDIR)/fv3config.yml
LOCAL_OUTDIR = rundir/nudge_mean_$*
RUNFILE = mean_nudging_runfile.py
GCS_OUTDIR = gs://vcm-ml-data/2020-03-30-learned-nudging-FV3GFS-runs/nudge_mean_$*
TEMPLATE = nudge_mean_template.yml

run_all_remote: run_remote_T run_remote_T_ps run_remote_T_ps_u_v

run_remote_%: prepare_$*
	python submit_to_kubernetes.py $(FV3CONFIG) $(GCS_OUTDIR)

run_local_%:
    fv3run --docker_image $(IMAGE) --runfile $(RUNFILE) $(FV3CONFIG) $(LOCAL_OUTDIR)

prepare_%:
	python prepare_config.py $(TEMPLATE) $* $(CONFIGDIR)

clean:
	rm -rf rundir configdir

.PHONY: run_all_remote
