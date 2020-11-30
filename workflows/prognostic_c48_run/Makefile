KEY_ARGS= -v $(GOOGLE_APPLICATION_CREDENTIALS):/key.json \
	  -e GOOGLE_APPLICATION_CREDENTIALS=/key.json
      
RUNDIR?=rundir
RESTARTS_PATH=gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts
INITIAL_CONDITION=20160801.001500
OUTPUT_ARGS=output-times-test.json
CONFIG_FILE=fv3config.yml
RUNFILE=sklearn_runfile.py
PREPARE_CONFIG_PY=prepare_config.py
RUN_FV3=runfv3.sh

NUDGING_BASE_CONFIG_FILE=nudging_config.yaml
NUDGING_CONFIG_FILE=nudging_config_temp.yaml
PROGNOSTIC_BASE_CONFIG_FILE=prognostic_config.yaml
PROGNOSTIC_CONFIG_FILE=prognostic_config_temp.yaml
MODEL=gs://vcm-ml-scratch/test-end-to-end-integration/integration-test-2c1ba84b1350/trained_model
NUDGE_TO_OBS_CONFIG_FILE=nudge_to_obs_config_temp.yaml
BASELINE_CONFIG_FILE=baseline_config_temp.yaml


build:
	docker-compose build fv3

dev:
	docker-compose run fv3 bash
    
fv3net_dev:
	docker-compose run $(KEY_ARGS) fv3net_dev bash

test:
	docker-compose run fv3 pytest

run_%: %_config_temp.yaml clean_rundir
	mv $*_config_temp.yaml $(CONFIG_FILE)
	$(RUN_FV3) $(CONFIG_FILE) $(RUNDIR) $(RUNFILE)

$(NUDGING_CONFIG_FILE): $(NUDGING_BASE_CONFIG_FILE)
	python3 $(PREPARE_CONFIG_PY) \
    $(NUDGING_BASE_CONFIG_FILE) \
    $(RESTARTS_PATH) \
    $(INITIAL_CONDITION) \
    --output-timestamps $(OUTPUT_ARGS) \
    > $(NUDGING_CONFIG_FILE)
    
$(PROGNOSTIC_CONFIG_FILE): $(PROGNOSTIC_BASE_CONFIG_FILE)
	python3 $(PREPARE_CONFIG_PY) \
    --model_url $(MODEL) \
    $(PROGNOSTIC_BASE_CONFIG_FILE) \
    $(RESTARTS_PATH) \
    $(INITIAL_CONDITION) \
    --output-timestamps $(OUTPUT_ARGS) \
    > $(PROGNOSTIC_CONFIG_FILE)
   
$(NUDGE_TO_OBS_CONFIG_FILE): $(NUDGE_TO_OBS_BASE_CONFIG_FILE)
	python $(PREPARE_CONFIG_PY) \
    --nudge-to-observations \
    $(PROGNOSTIC_BASE_CONFIG_FILE) \
    $(RESTARTS_PATH) \
    $(INITIAL_CONDITION) \
    --output-timestamps $(OUTPUT_ARGS) \
    > $(NUDGE_TO_OBS_CONFIG_FILE)
    
$(BASELINE_CONFIG_FILE): $(BASELINE_BASE_CONFIG_FILE)
	python prepare_config.py \
    $(PROGNOSTIC_BASE_CONFIG_FILE) \
    $(RESTARTS_PATH) \
    $(INITIAL_CONDITION) \
    --output-timestamps $(OUTPUT_ARGS) \
    > $(BASELINE_CONFIG_FILE)
    
clean_rundir:
	rm -rf $(RUNDIR) && mkdir $(RUNDIR)

.PHONY: dev test fv3net_dev clean_outdir run_