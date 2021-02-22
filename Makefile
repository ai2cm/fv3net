.PHONY: clean data lint

#################################################################################
# GLOBALS                                                                       #
#################################################################################
VERSION ?= $(shell git rev-parse HEAD)
REGISTRY ?= us.gcr.io/vcm-ml
ENVIRONMENT_SCRIPTS = .environment-scripts
PROJECT_NAME = fv3net
CACHE_TAG =latest

IMAGES = fv3net fv3fit post_process_run prognostic_run

.PHONY: build_images push_image run_integration_tests image_name_explicit
############################################################
# Docker Image Management
############################################################
# pattern rule for building docker images
build_image_%:
	tools/docker_build_cached.sh us.gcr.io/vcm-ml/$*:$(CACHE_TAG) \
		-f docker/$*/Dockerfile -t $(REGISTRY)/$*:$(VERSION) .

build_images: $(addprefix build_image_, $(IMAGES))
push_images: $(addprefix push_image_, $(IMAGES))

push_image_%: build_image_%
	docker push $(REGISTRY)/$*:$(VERSION)

pull_image_%:
	docker pull $(REGISTRY)/$*:$(VERSION)

build_image_ci:
	docker build -t us.gcr.io/vcm-ml/circleci-miniconda-gfortran:latest - < .circleci/dockerfile

############################################################
# Documentation (rules match "deploy_docs_%")
############################################################

## Empty rule for deploying docs
deploy_docs_%: 
	@echo "Nothing to do."


## Deploy documentation for fv3net to vulcanclimatemodeling.com
deploy_docs_fv3net:
	mkdir -p docs/_build/html
	docker run us.gcr.io/vcm-ml/fv3net:$(VERSION) tar -C docs/_build/html -c . | tar -C docs/_build/html -x
	gsutil -m rsync -R docs/_build/html gs://vulcanclimatemodeling-com-static/docs/fv3net

## Deploy documentation for prognostic run to vulcanclimatemodeling.com
deploy_docs_prognostic_run:
	mkdir html
	# use tar to grab docs from inside the docker image and extract them to "./html"
	docker run us.gcr.io/vcm-ml/prognostic_run tar -C docs/_build/html  -c . | tar -C html -x
	gsutil -m rsync -R html gs://vulcanclimatemodeling-com-static/docs/prognostic_c48_run
	rm -rf html
    
## Deploy documentation for fv3viz to vulcanclimatemodeling.com
deploy_docs_fv3viz:
	mkdir fv3viz_html
	# use tar to grab docs from inside the docker image and extract them to "./html"
	docker run us.gcr.io/vcm-ml/fv3net:$(VERSION) tar -C external/fv3fit/docs/_build/html  -c . | tar -C fv3viz_html -x
	gsutil -m rsync -R fv3viz_html gs://vulcanclimatemodeling-com-static/docs/fv3viz
	rm -rf fv3viz_html


############################################################
# Local Kubernetes
############################################################

## Install K8s and cluster manifests for local development
## Do not run for the GKE cluster
deploy_local:
	kubectl apply -f https://raw.githubusercontent.com/argoproj/argo/v2.11.6/manifests/install.yaml
	kubectl create secret generic gcp-key --from-file="${GOOGLE_APPLICATION_CREDENTIALS}"
	kubectl apply -f workflows/argo/cluster

############################################################
# Testing
############################################################
run_integration_tests:
	./tests/end_to_end_integration/run_test.sh $(REGISTRY) $(VERSION)

test_prognostic_run:
	docker run prognostic_run pytest

test_prognostic_run_report:
	bash workflows/prognostic_run_diags/tests/test_integration.sh

test_%:
	cd external/$* && tox

test_unit: test_fv3kube test_vcm
	coverage run -m pytest -m "not regression" --mpl --mpl-baseline-path=tests/baseline_images

test_regression:
	coverage run -m pytest -vv -m regression -s

test_dataflow:
	coverage run -m pytest -vv workflows/dataflow/tests/integration -s

coverage_report:
	coverage report -i --omit='**/test_*.py',conftest.py,'external/fv3config/**.py','external/fv3gfs-util/**.py','external/fv3gfs-wrapper/**.py','external/fv3gfs-fortran/**.py'

htmlcov:
	rm -rf $@
	coverage html -i --omit='**/test_*.py',conftest.py,'external/fv3config/**.py','external/fv3gfs-util/**.py''external/fv3gfs-wrapper/**.py','external/fv3gfs-fortran/**.py'

test_argo:
	make -C workflows/argo/ test

## Make Dataset
.PHONY: data update_submodules create_environment overwrite_baseline_images

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Set up python interpreter environment
update_submodules:
	git submodule sync --recursive
	git submodule update --recursive --init


overwrite_baseline_images:
	pytest tests/test_diagnostics_plots.py --mpl-generate-path tests/baseline_images


############################################################
# Dependency Management
############################################################

lock_deps: lock_pip
	conda-lock -f environment.yml
	# external directories must be explicitly listed to avoid model requirements files which use locked versions

.PHONY: lock_pip
lock_pip:
	pip-compile  \
	--no-annotate \
	external/vcm/setup.py \
	pip-requirements.txt \
	external/fv3fit/requirements.txt \
	workflows/post_process_run/requirements.txt \
	workflows/prognostic_run_diags/requirements.txt \
	docker/**/requirements.txt \
	--output-file constraints.txt
	# remove extras in name: e.g. apache-beam[gcp] --> apache-beam
	sed -i.bak  's/\[.*\]//g' constraints.txt
	rm -f constraints.txt.bak

## Install External Dependencies
install_deps:
	bash $(ENVIRONMENT_SCRIPTS)/build_environment.sh $(PROJECT_NAME)

## Install Local Packages for the "fv3net" environment
install_local_packages:
	bash $(ENVIRONMENT_SCRIPTS)/install_local_packages.sh $(PROJECT_NAME)

create_environment:
	bash $(ENVIRONMENT_SCRIPTS)/build_environment.sh $(PROJECT_NAME)
	bash $(ENVIRONMENT_SCRIPTS)/install_local_packages.sh $(PROJECT_NAME)

############################################################
# Linting
############################################################

PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')

check_file_size:
	./tests/check_for_large_files.sh

typecheck:
	./check_types.sh

lint: check_file_size
	black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	flake8 --ignore=F401 E203 $(PYTHON_INIT_FILES)
	@echo "LINTING SUCCESSFUL"

reformat:
	black $(PYTHON_FILES) $(PYTHON_INIT_FILES)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
