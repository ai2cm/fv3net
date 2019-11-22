#!/bin/bash

BEAM_MAIN_MODULE=fv3net.pipelines.coarsen_surface_c3072

# submit the job
python -m $BEAM_MAIN_MODULE
