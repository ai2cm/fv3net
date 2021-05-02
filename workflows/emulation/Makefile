SUBSAMPLE_CONFIG ?=

subsample:
	python subset_training.py ${SUBSAMPLE_CONFIG}

subsample_hybrid_train: SUBSAMPLE_CONFIG = simple_hybridedmf_train.yaml
subsample_hybrid_train: subsample

subsample_hybrid_valid: SUBSAMPLE_CONFIG = simple_hybridedmf_valid.yaml
subsample_hybrid_valid: subsample