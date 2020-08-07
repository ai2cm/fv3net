This workflow generates diagnostic outputs for variables predicted by 
a trained ML model. Currently, the workflow only accepts models wrapped 
for usage with xarray dataset inputs and outputs (the `SklearnWrapper` class 
found in `fv3fit.sklearn.wrapper`.) 

The script takes in a configuration YAML file (which contains the input data path), 
a trained ML model, and the output path. An optional json file containing a list 
of timesteps to use can also be provided. If not provided, the workflow will use 
all timesteps present in the data.
```
python -m offline_ml_diags.compute_diags \
    $CONFIG_YAML \
    $MODEL \
    $OUTPUT \
    --timesteps-file $TIMESTEP_LIST_JSON
```

The cosine zenith angle feature is a special case of a feature variable that is not
present in the dataset and must be derived after the mapper reads the data. To include it
as a feature, provide the `model_mapper_kwarg` mapping `cos_z_var: <name of cosine z feature>`
in the configuration. An example is below.


Example config:
```
variables:
  - air_temperature
  - specific_humidity
  - dQ1
  - dQ2
  - pQ1
  - pQ2
  - pressure_thickness_of_atmospheric_layer
  - land_sea_mask
  - surface_geopotential
model_mapper_kwargs:
  cos_z_var: cos_zenith_angle
mapping_function: open_fine_resolution_nudging_hybrid
mapping_kwargs:
  nudging:
    shield_diags_url: gs://vcm-ml-experiments/2020-06-17-triad-round-1/coarsen-c384-diagnostics/coarsen_diagnostics/gfsphysics_15min_coarse.zarr
    offset_seconds: -900
    nudging_url: gs://vcm-ml-experiments/2020-06-30-triad-round-2/hybrid/nudging
  fine_res:
    offset_seconds: 450
    fine_res_url: gs://vcm-ml-experiments/2020-06-02-fine-res/fine_res_budget      
batch_kwargs:
  timesteps_per_batch: 10
data_path: this_isnt_used
```


Example usage (from top level of `fv3net`): 
```
python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/config.yml \
    gs://vcm-ml-scratch/andrep/test-nudging-workflow/train_sklearn_model/sklearn_model.pkl \
    gs://vcm-ml-scratch/annak/test-offline-validation-workflow \
    --timesteps-file workflows/offline_ml_diags/tests/times.json
```

#### Creating reports
Report HTMLs may be created using `offline_ml_diags.create_report`, where the input data path should be
the output path of the `offline_ml_diags.compute_diags` script. The output location can be either a local
or remote GCS directory.

Example usage:
```
python -m offline_ml_diags.create_report \
    $INPUT \
    $OUTPUT 
```