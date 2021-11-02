TRAIN_DATA_CONFIG?=train-data.yaml
VAL_DATA_CONFIG?=test-data.yaml
TRAIN_CONFIG?=locally-connected-training-config.yaml
MODEL_DIR?=model
DIAGS_DIR?=offline_diags
REPORT_DIR?=offline_report

train:
	python3 -m fv3fit.train $(TRAIN_CONFIG) $(TRAIN_DATA_CONFIG) $(MODEL_DIR) --validation-data-config $(VAL_DATA_CONFIG)

offline_diags:
	python3 -m fv3net.diagnostics.offline.compute $(MODEL_DIR) $(VAL_DATA_CONFIG) $(DIAGS_DIR)

offline_report: offline_diags
	python3 -m fv3net.diagnostics.offline.create_report $(DIAGS_DIR) $(REPORT_DIR)

.PHONY: train offline_diags offline_report
