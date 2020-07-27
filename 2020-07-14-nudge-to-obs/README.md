### Workflow configuration for nudge-to-obs ML

There are two parts to this workflow which must be manually submitted.

First:
```
    make submit_nudge_to_obs
```
to do a 40-day run that is nudged towards GFS analysis. This should take about 4 hours including post-processing and uploading of data.

Then:
```
    make submit_train_and_prognostic
```
which will train an ML-model on the nudging tendencies from the preceding run and do two prognostic runs (one 10-day, and one 35-day).
