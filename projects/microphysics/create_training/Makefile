
CONFIG ?= ./training_fv3config_template.yaml

create:
	argo submit argo.yaml \
		-p config="$(shell base64 --wrap=0 $(CONFIG))" \

gather:
	python ../scripts/gather_netcdfs.py \
		gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/create-training-microphysics-v4-2- \
		gs://vcm-ml-experiments/microphysics-emulation/2022-04-18/microphysics-training-data-v4
