# Surface coarsening workflow

This workflow coarse-grains the surface diagnostic output of a SHIELD C3072 simulation.

This workflow produced the outputs at
`gs://vcm-ml-data/2019-11-06-X-SHiELD-gfsphysics-diagnostics-coarsened/C384/`.

To test this job locally, run 
    
    bash submit_local.sh

To submit with many workers on google cloud dataflow run

    bash submit_job.sh
    
This script can be edited to change things like the number of workers to use in
dataflow. It serves as a good template for future dataflow pipelines.
