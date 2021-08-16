build_image:
	docker build -t us.gcr.io/vcm-ml/circleci-miniconda3-gfortran .

push_image: build_image
	docker push us.gcr.io/vcm-ml/circleci-miniconda3-gfortran

.PHONY: build_image push_image
