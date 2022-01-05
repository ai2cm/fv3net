REPORT_PATH=vcm-ml-public/microphysics-emulation/$(shell git rev-parse HEAD)

reports = introduction.html error-scale.html

upload: $(reports) index.html
	gsutil cp -r $^ gs://$(REPORT_PATH)
	@echo "reports at https://storage.googleapis.com/$(REPORT_PATH)/index.html"

.PHONY: upload

index.html: $(reports)
	bash make_index.sh $^

.PHONY: index.html

%.ipynb: %.py
	jupytext --execute --to .ipynb $<

%.ipynb: %.md
	jupytext --to .ipynb $<

%.html: %.ipynb
	jupyter nbconvert --to html $<
