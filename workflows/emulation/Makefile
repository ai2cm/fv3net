SUBSAMPLE_CONFIG ?=
TRAINING_CONFIG ?=
SUBSET_DESTINATION ?= /tmp/subset_training_data

subsample:
	python subset_training.py ${SUBSAMPLE_CONFIG}

subsample_hybrid_train: SUBSAMPLE_CONFIG = simple_hybridedmf_train.yaml
subsample_hybrid_train: subsample

subsample_hybrid_valid: SUBSAMPLE_CONFIG = simple_hybridedmf_valid.yaml
subsample_hybrid_valid: subsample

get_subsampled_training: SUBSET_SOURCE = gs://vcm-ml-scratch/andrep/all-physics-emu/training-subsets/simple-phys-hybridedmf-10day/
get_subsampled_training:
	mkdir -p ${SUBSET_DESTINATION}/training/
	gsutil -m cp -r ${SUBSET_SOURCE} ${SUBSET_DESTINATION}/training/

get_subsampled_testing: SUBSET_SOURCE = gs://vcm-ml-scratch/andrep/all-physics-emu/validation-subsets/simple-phys-hybridedmf-10day/
get_subsampled_testing:
	mkdir -p ${SUBSET_DESTINATION}/testing/
	gsutil -m cp -r ${SUBSET_SOURCE} ${SUBSET_DESTINATION}/testing/

get_subsampled: get_subsampled_training get_subsampled_testing

train_emulator: TRAINING_CONFIG = training_config.yaml
train_emulator:
	python train_emulator.py ${TRAINING_CONFIG}

.PHONY: train_emulator