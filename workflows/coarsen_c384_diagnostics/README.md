#### Usage
Run 
`workflows/coarsen_c384_diagnostics/coarsen_c384_diagnostics.sh`
`  {INPUT_LOCATION} {CONFIG_LOCATION} {OUTPUT_LOCATION}`

#### Description
This script coarsens a subset of diagnostic variables from the high resolution runs
and saves them as a zarr in a coarsened resolution. Input data must be in the form
of coarsened C384 SHiELD diagnostic .zarrs. The input and output locations
may be local or remote paths. An example config YAML file is provided:
`coarsen-c384-diagnostics.yml`

The output of this workflow is used later
as an input to the training and testing data pipeline steps. In the future we 
can move this task into the SHiELD post-processing and upload workflow.