
CONFIG ?= ./training_fv3config_template.yaml
RUN_DATE ?= 2021-11-18

create:
	argo submit argo.yaml \
		-p config="$(shell base64 --wrap=0 $(CONFIG))" \

gather:
	python ../scripts/gather_netcdfs.py $(RUN_DATE) $(TAG_PREFIX)
