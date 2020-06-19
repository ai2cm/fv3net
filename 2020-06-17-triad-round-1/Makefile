
check:
	kubectl apply -f time_configmap.yaml --dry-run=client
	kubectl apply -k nudging --dry-run=client
	kubectl apply -k one-step-clouds-off --dry-run=client
	kubectl apply -k one-step-physics-off --dry-run=client
	kubectl apply -f fine-res/ --dry-run=client

submit:
	# this needs to be done first
	kubectl apply -f time_configmap.yaml
	kubectl apply -k nudging
	kubectl apply -k one-step-clouds-off
	kubectl apply -k one-step-physics-off
	kubectl apply -f fine-res/

delete:
	kubectl delete -f time_configmap.yaml || exit 0
	kubectl delete -k nudging || exit 0
	kubectl delete -k one-step-clouds-off || exit 0
	kubectl delete -k one-step-physics-off || exit 0
	kubectl delete -f fine-res/ || exit 0

