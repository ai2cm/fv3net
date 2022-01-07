REPORT_PATH=vcm-ml-public/microphysics-emulation/$(shell git rev-parse HEAD)

reports =  introduction.html training-data.html error-scale.html debug_log_transform.html

upload: build
	gsutil -m cp -r _build/html/ gs://$(REPORT_PATH)
	@echo "reports at https://storage.googleapis.com/$(REPORT_PATH)/index.html"

server:
	cd _build/html && python -m http.server 32341

.PHONY: upload

.PHONY: build
build: 
	jupyter-book build .
