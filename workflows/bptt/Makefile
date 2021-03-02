
PROGNOSTIC_RUN_IMAGE=us.gcr.io/vcm-ml/prognostic_run

build:
	docker-compose build fv3fit prognostic_run

dev:
	docker-compose run fv3fit bash

dev_prognostic_run:
	docker-compose run prognostic_run bash

arrays:
	python3 preprocessing.py arrays

model:
	python3 train.py arrays model

offline_diags: 
	python -m offline_ml_diags.compute_diags model offline_diags --config-yml offline_diags.yaml

offline_report:
	$(RM) -r offline_report
	python -m offline_ml_diags.create_report offline_diags offline_report

fv3config.yaml: prognostic_base.yaml
	python3 ../prognostic_c48_run/prepare_config.py \
		prognostic_base.yaml \
		gs://vcm-ml-code-testing-data/c48-restarts-for-e2e \
		20160801.001500 \
		--model_url $(shell pwd)/model \
    	> fv3config.yaml

prognostic_run: fv3config.yaml
	runfv3 create prognostic_run fv3config.yaml ../prognostic_c48_run/sklearn_runfile.py
	runfv3 append prognostic_run

.PHONY: dev test arrays model offline_diags offline_report prognostic_run
