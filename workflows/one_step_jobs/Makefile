root_url = gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs

all-physics-off:
	python submit_jobs.py \
	    --one-step-yaml all-physics-off.yml \
        --input-url $(root_url)/restarts/C48 \
        --output-url $(root_url)/one-step-all-physics-off/C48

deep-and-mp-off:
	python submit_jobs.py \
	    --one-step-yaml deep-and-mp-off.yml \
        --input-url $(root_url)/restarts/C48 \
        --output-url $(root_url)/one-step-deep-and-mp-off/C48

test-submission:
	python submit_jobs.py \
	    --one-step-yaml all-physics-off.yml \
        --input-url $(root_url)/restarts/C48 \
        --output-url gs://vcm-ml-data/test-oliwm/one-step-all-physics-off \
        --n-steps 5