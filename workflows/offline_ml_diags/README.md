This workflow generates diagnostic outputs for variables predicted by 
a trained ML model. Currently, the workflow only accepts models wrapped 
for usage with xarray dataset inputs and outputs (the `SklearnWrapper` class 
found in `fv3net.regression.sklearn.wrapper`.) 

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

Example config:
```
data_path: gs://vcm-ml-scratch/andrep/test-nudging-workflow/nudging
rename_variables:
  air_temperature_tendency_due_to_nudging: dQ1
  specific_humidity_tendency_due_to_nudging: dQ2
variables:
  - air_temperature
  - specific_humidity
  - dQ1
  - dQ2
  - pressure_thickness_of_atmospheric_layer
mapping_function: open_nudged
mapping_kwargs:
  nudging_timescale_hr: 3
  initial_time_skip_hr: 0
batch_kwargs:
  timesteps_per_batch: 5
  init_time_dim_name: "initial_time"
```


Example usage (from top level of `fv3net`): 
```
python -m offline_ml_diags.compute_diags \
    workflows/offline_ml_diags/tests/config.yml \
    gs://vcm-ml-scratch/andrep/test-nudging-workflow/train_sklearn_model/sklearn_model.pkl \
    gs://vcm-ml-scratch/annak/test-offline-validation-workflow \
    --timesteps-file workflows/offline_ml_diags/tests/times.json
```