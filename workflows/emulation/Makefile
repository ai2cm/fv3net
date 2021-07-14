SUBSAMPLE_CONFIG ?=
TRAINING_CONFIG ?=
SUBSET_DESTINATION ?= /tmp/subset_training_data
SUBSET_SOURCE ?= gs://vcm-ml-scratch/andrep/all-physics-emu/validation-subsets/

subsample:
	python subset/subset_training_data.py ${SUBSAMPLE_CONFIG}

subsample_hybrid_train: SUBSAMPLE_CONFIG = subset/configs/simple_hybridedmf_train.yaml
subsample_hybrid_train: subsample

subsample_hybrid_valid: SUBSAMPLE_CONFIG = subset/configs/simple_hybridedmf_valid.yaml
subsample_hybrid_valid: subsample

get_subsampled_training: SUBSET_SOURCE = gs://vcm-ml-experiments/andrep/2021-05-28/training-subsets/
get_subsampled_training:
	mkdir -p ${SUBSET_DESTINATION}/training/
	gsutil -m cp -r ${SUBSET_SOURCE} ${SUBSET_DESTINATION}/training/

get_subsampled_testing: SUBSET_SOURCE = gs://vcm-ml-experiments/andrep/2021-05-28/validation-subsets/
get_subsampled_testing:
	mkdir -p ${SUBSET_DESTINATION}/testing/
	gsutil -m cp -r ${SUBSET_SOURCE} ${SUBSET_DESTINATION}/testing/

get_subsampled: get_subsampled_training get_subsampled_testing

train_emulator: TRAINING_CONFIG = training_config.yaml
train_emulator:
	python3 train_emulator.py ${TRAINING_CONFIG}

.PHONY: train_emulator