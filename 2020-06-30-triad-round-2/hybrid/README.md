# Fine resolution with nudging-derived baseline

## Nudging run + basic features

To submit the job to argo, run

    make submit

The timesteps for nudging and training are inserted by the `prepare_config.sh`
script into the file `config.json` which is then passed to argo. To create this
file without submitting, run

    make config.json

This is run automatically as part of the submit rule.

## Extra features experiment

This can be submitted with

    make submit_extra_vars

## Requirements

- jq v1.6
- [yq](https://github.com/kislyuk/yq) v2.10.0
- bash v5
- argo v2.8.1
