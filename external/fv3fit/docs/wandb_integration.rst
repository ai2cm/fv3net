Integration with wandb
======================


Running `fv3fit.train` with the optionial flag `--wandb` will log the training run with wandb.
See https://docs.wandb.ai/quickstart for instructions on setting up wandb.

Wandb automatically initialized using the environment variables `WANDB_ENTITY, WANDB_PROJECT, WANDB_JOB_TYPE`
if they are set. Change these environment variables if you want to change where wandb will log runs. See
for more detals: https://docs.wandb.ai/guides/track/advanced/environment-variables.


