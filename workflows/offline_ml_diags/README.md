# Offline ML Diagnostics workflow

This workflow generate offline diagnostics datasets, $R^2$, and bias metrics
for a given ML model. It contains two steps: 1) `compute_diags`, which calculates the
aforementioned data, and 2) `create_report`, which compiles an HTML report of tables and
figures and uploads it to a destination. If that destination is a public bucket, the report
can be viewed at its URL on a web browser.
## Quickstart

### 1. Generating diagnostics and metrics
Generating the diagnostics can be done by providing a path to a model trained and dumped by 
`fv3fit`, an output path, and a flag specifiying how to select the test set of timesteps.
e.g.,
```
python -m offline_ml_diags.compute_diags \
    $MODEL_PATH \
    $DIAGNOSTICS_OUTPUT_PATH \
    --timesteps-file $TIMESTEP_LIST_JSON
```

#### Specifying the test set
The test set of timesteps may provided via one of the following options:
- `--timesteps-file $TIMES_JSON` : Provide a JSON file with a list of timesteps 
(`"YYYYMMDD.HHMMSS"` formatted strings)
- `--num-test-sample $N` : Sample `N` timesteps from outside the training dataset
range of times
- `--training` (just the flag, no arg provided): Use the set of training timesteps, read
from the training configuration saved with the model. This a safety flag to prevent accidental 
evaluation of skill on the training set.
- `--config_yml $CONFIG` : Use a different config file from the trained model, which 
may contain a separate set of timesteps. *If no `batch_kwargs["timesteps"]` field is present in the config file provided, all timesteps in the dataset mapper are used. This can be a very large number of timesteps*.


### 2. Generating an HTML report
The HTML report is generate by providing the path to the diagnostics and metrics
calculated in the previous step, and the desired output path.
```
python -m offline_ml_diags.create_report \
    $DIAGNOSTICS_OUTPUT_PATH \
    $REPORT_OUTPUT_PATH
```

## Generating diagnostics and metrics

### Inputs
#### Optional: User provided configuration file
The simplest way to specify the test dataset to use is to use the configuration saved
in the trained model; in this case you do not need to be concerned with providing the
following information in a configuration file.

If a separate configuration YAML file is used via the optional `--config-yml` flag, it must provide
the following information as top level keys:
- `batch_function`: must be `batches_from_geodata`
- `batch_kwargs`: kwargs passed to the batch loading function `loaders.<batch_function>`
specified above, see [loaders]<link to loaders docs> for details.
- `input_variables`: variable names of ML model features
- `output_variables`: ML model target variables
- (optional) `data_path`: location of the test dataset; if not present in the configuration
file it must be provided via the command line arg `--data-path`

### Outputs
This workflow generates the following outputs:
- Copy of the config used to generate the diagnostics
- Distribution of the timesteps in test set, both day and hour
- Zonal average of bias and $R^2$ at each pressure level
- Average bias, RMSE, and $R^2$ at each pressure level
- Average vertical profile of predicted variables, for global, land, ocean,
positive net precipitation, and negative net precipitation domains
- Time averaged maps of column integrated predicted variables
- Diurnal cycle of column integrated dQ1 and dQ2, for global, land, and ocean domains
- Single timestep snapshot of predicted vs. target varible, along a 0 deg longitude transect
- Model's Jacobian evaluated at the input means (only for neural net models) 


## CLI and full list of arguments

### `offline_ml_diags.compute_diags`

```bash
usage: python -m offline_ml_diags.compute_diags [-h] [--data-path [DATA_PATH [DATA_PATH ...]]]
               [--config-yml CONFIG_YML] [--timesteps-file TIMESTEPS_FILE]
               [--snapshot-time SNAPSHOT_TIME]
               [--timesteps-n-samples TIMESTEPS_N_SAMPLES] [--training]
               [--grid GRID]
               model_path output_path

```

|arg|default|help|
| :--- | :--- | :--- |
|`model_path`||Local or remote path for reading ML model.|
|`output_path`||Local or remote path where diagnostic output will be written.|
|`--help`||show this help message and exit|
|`--data-path`|`None`|Location of test data. If not provided, will use the data_path saved with the trained model config file.|
|`--config-yml`|`None`|Config file with dataset and variable specifications.|
|`--timesteps-file`|`None`|Json file that defines train timestep set. Overrides any timestep set in training config if both are provided.|
|`--snapshot-time`|`None`|Timestep to use for snapshot. Provide a string 'YYYYMMDD.HHMMSS'. If provided, will use the closest timestep in the test set. If not, will default to use the first timestep available.|
|`--timesteps-n-samples`|`None`|If specified, will draw attempt to draw this many test timesteps from either i) the mapper keys that lie outside the range of times in the config timesteps or ii) the set of timesteps provided in --timesteps-file.Random seed for sampling is fixed to 0. If there are not enough timesteps available outside the config range, will return all timesteps outside the range. Useful if args.config_yml is taken directly from the trained model.Incompatible with also providing a timesteps-file arg. |
|`--training`|False|If provided, allows the use of timesteps from the trained model config to be used for offline diags. Only relevant if no config file is provided and no optional args for timesteps-file or timesteps-n-samples given. Acts as a safety to prevent accidental use of training set for the offline metrics.|
|`--grid`|`None`|Optional path to grid data netcdf. If not provided, defaults to loading the grid  with the appropriate resolution (given in batch_kwargs) from the catalog. Useful if you do not have permissions to access the GCS data in vcm.catalog.|

### `offline_ml_diags.create_report`
```bash
usage: python -m offline_ml_diags.create_report [-h] [--commit-sha COMMIT_SHA] input_path output_path

```
|arg|default|help|
| :--- | :--- | :--- |
|`--help`||show this help message and exit|
|`--commit-sha`|`None`|Commit SHA of fv3net used to create report. Useful for referencingthe version used to train the model.|

