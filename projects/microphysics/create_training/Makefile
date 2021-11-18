
CONFIG ?= ./training_fv3config_template.yaml
OUTPUT_FREQUENCY ?= 18000
TAG_PREFIX ?= create-training
RUN_DATE ?= 2021-11-18

GCS_BASE = gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper

create:
	./run_single.sh 01 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 02 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 03 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 04 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 05 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 06 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 07 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 08 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 09 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 10 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 11 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)
	./run_single.sh 12 $(CONFIG) $(OUTPUT_FREQUENCY) $(TAG_PREFIX)

gather:
	python ../scripts/gather_netcdfs.py $(RUN_DATE) $(TAG_PREFIX)
