python -m offline_ml_diags.compute_diags \
    gs://vcm-ml-experiments/2020-10-30-nudge-to-obs-GRL-paper/nudge-to-obs-run-3hr-diags \
    training_config.yml \
    gs://vcm-ml-experiments/2020-12-03-sensitivity-to-amount-of-input-data/neural-network-128-timesteps-1-epochs-seed-0/trained_model \
    output
