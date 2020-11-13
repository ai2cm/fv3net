OUTDIR?=outdir
BASE_CONFIG_FILE=./nudging_config.yaml
RESTARTS_PATH=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts
INITIAL_CONDITION="20160801.001500"
CONFIG_FILE=./fv3config.yml
RUNFILE=./nudging_runfile.py
PREPARE_CONFIG_PY=../prepare_config.py
RUN_FV3=../runfv3.sh

all: run

run: $(CONFIG_FILE)
	$(RUN_FV3) $(CONFIG_FILE) $(OUTDIR) $(RUNFILE)

$(CONFIG_FILE): $(BASE_CONFIG_FILE)
	python3 $(PREPARE_CONFIG_PY) \
    $(BASE_CONFIG_FILE) \
    $(RESTARTS_PATH) \
    $(INITIAL_CONDITION) > $(CONFIG_FILE)

.PHONY: run
