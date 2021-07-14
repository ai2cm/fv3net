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

Data can be configured using a yaml that conforms to `loaders.BatchesLoader`. Snapshot data such as transect plots requires you provide a `loaders.BatchesFromMapper`. This configuration path is passed to the `data_yaml` command-line argument.

#### Outputs
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


### 2. Generating an HTML report
The HTML report is generate by providing the path to the diagnostics and metrics
calculated in the previous step, and the desired output path.
```
python -m offline_ml_diags.create_report \
    $DIAGNOSTICS_OUTPUT_PATH \
    $REPORT_OUTPUT_PATH
```

## CLI and full list of arguments

### `offline_ml_diags.compute_diags`

```bash
$ python -m offline_ml_diags.compute_diags --help

usage: compute_diags.py [-h] [--snapshot-time SNAPSHOT_TIME] [--grid GRID]
                        model_path output_path data_yaml

positional arguments:
  model_path            Local or remote path for reading ML model.
  output_path           Local or remote path where diagnostic output will be
                        written.
  data_yaml             Config file with dataset specifications.

optional arguments:
  -h, --help            show this help message and exit
  --snapshot-time SNAPSHOT_TIME
                        Timestep to use for snapshot. Provide a string
                        'YYYYMMDD.HHMMSS'. If provided, will use the closest
                        timestep in the test set. If not, will default to use
                        the first timestep available.
  --grid GRID           Optional path to grid data netcdf. If not provided,
                        defaults to loading the grid with the appropriate
                        resolution (given in batch_kwargs) from the catalog.
                        Useful if you do not have permissions to access the
                        GCS data in vcm.catalog.
```

### `offline_ml_diags.create_report`
```bash
usage: python -m offline_ml_diags.create_report [-h] [--commit-sha COMMIT_SHA] input_path output_path

```
|arg|default|help|
| :--- | :--- | :--- |
|`--help`||show this help message and exit|
|`--commit-sha`|`None`|Commit SHA of fv3net used to create report. Useful for referencingthe version used to train the model.|

