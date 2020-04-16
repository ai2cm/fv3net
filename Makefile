GCR_BASE  = us.gcr.io/vcm-ml
FV3NET_IMAGE = $(GCR_BASE)/fv3net:b8143681219169cdd8ad2c47aa53ffe5dc664364
PROGNOSTIC_RUN_IMAGE = $(GCR_BASE)/prognostic_run:b8143681219169cdd8ad2c47aa53ffe5dc664364

versions:
	@echo "jq version info"
	jq --version
	@echo "------"
	@echo "bash:"
	bash --version
	@echo "------"
	@echo "kubectl:"
	kubectl version
	@echo "------"
	@echo "gettext:"
	gettext --version

generate_configs:
	end_to_end/generate_all_configs.sh $(PROGNOSTIC_RUN_IMAGE) $(FV3NET_IMAGE) 

submit:
	kubectl apply -f manifests/
	kubectl apply -f jobs/
