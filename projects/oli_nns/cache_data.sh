#!/usr/bin/env bash

set -e -x

variable_names="cos_zenith_angle air_temperature specific_humidity land_sea_mask surface_geopotential latent_heat_flux sensible_heat_flux Q1 Q2 air_temperature_tendency_due_to_nudging specific_humidity_tendency_due_to_nudging latitude longitude vertical_wind surface_temperature surface_precipitation_rate latent_heat_flux sensible_heat_flux t_dt_phys_coarse qv_dt_phys_coarse dq3dt_deep_conv_coarse dq3dt_shal_conv_coarse eddy_flux_vulcan_omega_sphum"

python -m loaders.batches.save train.yaml download/cached_train --variable-names $variable_names
python -m loaders.batches.save validation.yaml download/cached_validation --variable-names $variable_names
# python -m loaders.batches.save test.yaml cached_test --variable-names $variable_names
