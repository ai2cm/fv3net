EXPERIMENTS = physics-off clouds-off deep-off physics-on

run_all:
	for exp in $(EXPERIMENTS) ; do \
		kubectl apply -k $$exp ; \
	done ; \

run_%:
	kubectl apply -k $*

check:
	for exp in $(EXPERIMENTS) ; do \
		kubectl apply -k $$exp --dry-run -o yaml > /dev/null; \
	done ; \
	echo "Kustomize configs are ok."


