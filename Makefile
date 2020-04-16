
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
	./end_to_end/generate_all_configs.sh

submit:
	kubectl apply -f manifests/
