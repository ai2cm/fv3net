IMAGE=us.gcr.io/vcm-ml/regrid_c3072_diag

all: pipeline

push: build
	docker push $(IMAGE)

build:
	docker build -t $(IMAGE) .

pipeline: 
	dsl-compile --py make_pipeline.py --output regrid_c3072_diag.tar.gz
