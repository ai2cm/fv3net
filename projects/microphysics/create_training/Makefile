
CONFIG ?= ./training_fv3config_template.yaml
OUTPUT_FREQUENCY ?= 18000


create_training:
	./run_single.sh $(CONFIG) 01 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 02 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 03 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 04 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 05 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 06 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 07 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 08 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 09 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 10 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 11 $(OUTPUT_FREQUENCY)
	./run_single.sh $(CONFIG) 12 $(OUTPUT_FREQUENCY)


delete:
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160101.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160201.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160301.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160401.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160501.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160601.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160701.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160801.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20160901.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20161001.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20161101.000000
	gsutil -m rm -r gs://vcm-ml-scratch/2021-10-14-microphysics-emulation-paper/2021-11-18/test-create-training-20161201.000000