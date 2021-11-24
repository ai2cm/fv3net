
CONFIG ?= ./training_fv3config_template.yaml
OUTPUT_FREQUENCY ?= 18000
TAG_PREFIX ?= microphysics-training-data-v2
RUN_DATE ?= 2021-11-18

create:
	argo submit argo.yaml \
		-p config="$(shell base64 --wrap=0 $(CONFIG))" \
		-p tag="${TAG_PREFIX}" \
		-p output_frequency=${OUTPUT_FREQUENCY}

gather:
	python ../scripts/gather_netcdfs.py $(RUN_DATE) $(TAG_PREFIX)
