BASE_TIME_CONFIG = ../2020-06-17-triad-round-1/time_configmap.yaml

check:
	kubectl apply -f $(BASE_TIME_CONFIG) --dry-run=client
	kubectl apply -k nudging-additional-features --dry-run=client


submit_nudging_additional_features:
	# this needs to be done first
	kubectl apply -f $(BASE_TIME_CONFIG)
	kubectl apply -k nudging-additional-features


delete:
	kubectl delete -f $(BASE_TIME_CONFIG) || exit 0
	kubectl delete -k nudging-additional-features || exit 0