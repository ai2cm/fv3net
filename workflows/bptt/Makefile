
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

evaluate:
	python3 evaluate.py arrays model

offline_diags: 
	python -m offline_ml_diags.compute_diags model offline_diags --config-yml offline_diags.yaml

offline_report:
	$(RM) -r offline_report
	python -m offline_ml_diags.create_report offline_diags offline_report

fv3config.yaml: prognostic_base.yaml
	python3 ../prognostic_c48_run/prepare_config.py \
		prognostic_base.yaml \
		gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts \
		20160805.000000 \
		--model_url $(shell pwd)/model \
		--model_url gs://vcm-ml-experiments/2021-01-26-c3072-nn/l2/winds-seed-0/trained_model \
    	> fv3config.yaml

prognostic_run: fv3config.yaml
	runfv3 create prognostic_run fv3config.yaml ../prognostic_c48_run/sklearn_runfile.py
	runfv3 append prognostic_run

.PHONY: dev test arrays model evaluate offline_diags offline_report prognostic_run
