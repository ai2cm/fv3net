RUNFILE ?= runfile_recurrent.py
BASE_CONFIG_FILE ?= nudging_config.yml
RUNDIR ?= rundir
TMP_CONFIG_FILE = processed_config.yml
PREPARE_CONFIG_PY = prepare_config.py
FV3_DATA_DIR = "/Volumes/Extreme SSD/vcm-ml-data/"


build:
	docker-compose build train_bptt

dev:
	FV3_DATA_DIR=$(FV3_DATA_DIR) docker-compose run train_bptt bash

$(RUNDIR)/input.nml:
	python3 $(PREPARE_CONFIG_PY) $(BASE_CONFIG_FILE) > $(TMP_CONFIG_FILE)
	write_run_directory $(TMP_CONFIG_FILE) $(RUNDIR)
	#$(RM) $(TMP_CONFIG_FILE)

clean:
	$(RM) -r $(RUNDIR)

$(CONFIG_FILE): $(BASE_CONFIG_FILE)
	python3 $(PREPARE_CONFIG_PY) $(BASE_CONFIG_FILE) > $(CONFIG_FILE)

run: $(RUNDIR)/input.nml
	cp $(RUNFILE) $(RUNDIR)/$(RUNFILE)
	cd $(RUNDIR) &&	mpirun -n 6 --allow-run-as-root --mca btl_vader_single_copy_mechanism none python3 -m mpi4py $(RUNFILE)

.PHONY: build dev
