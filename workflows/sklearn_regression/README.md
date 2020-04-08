## Training and testing sklearn models

### Training
To train a sklearn regression model, create a training configuration yaml and execute 
the script `train_sklearn.sh`. The script will create a timestamped directory and save
the trained model output as well as a copy of the model configuration.

Example shell script:
```
python -m fv3net.regression.sklearn.train \
  gs://vcm-ml-data/test_annak/2020-02-05_train_data_pipeline #input data path where "train" folder is located
  example_rf_training_config.yml \
  {output_data_path} \
  --delete-local-results-after-upload True
```
The last two arguments are optional and allow the user to save the output directory to 
a remote storage location instead of a local directory.

Example model configuration YAML:
```
model_type: sklearn_random_forest
hyperparameters:
  min_samples_leaf: 8
  n_estimators: 4   # number of estimators to train FOR EACH BATCH
num_batches: 3  # number of batches to train
files_per_batch: 5  # number of input zarr files to concatenate for each train batch
input_variables:  # features to use in prediction
  - T
  - sphum
  - phis
  - slmsk
  - tsea
  - insolation
  - SHF
  - LHF
output_variables:  # variables to predict
  - Q1
  - Q2

```
Currently, only random forests are implemented for training sklearn models in fv3net.
Note that the total number of trees in the random forest will be 
`hyperparameters.n_estimators * num_batches`.


### Testing
The script `test_sklearn.sh` will use a trained model to predict diagnostics on a test
dataset. The resulting figures as well as a consolidated html report will be saved to
a timestamped output directory. 

Example shell script:
```
python -m fv3net.regression.model_diagnostics \
  gs://vcm-ml-data/test-annak/2020-02-05_train_data_pipeline \ # location of "test" directory
  20200205.205016_model_training_files \ # location of "sklearn_model.pkl" file
  gs://vcm-ml-data/2019-12-05-40-day-X-SHiELD-simulation-C384-diagnostics/C48_gfsphysics_15min_coarse.zarr \
  {output_path} \
  --num-test-zarrs 8
```