check:
	kubectl apply -f time_configmap.yaml --dry-run=client
	kubectl apply -k nudging-additional-features --dry-run=client


submit_nudging_additional_features:
	# this needs to be done first
	kubectl apply -f time_configmap.yaml
	kubectl apply -k nudging-additional-features


delete:
	kubectl delete -f time_configmap.yaml || exit 0
	kubectl delete -k nudging-additional-features || exit 0