# wandb-query

Command line tool for reading runs from wandb

Composable with other command tools.

Listing some runs
```
$ wandb-query | head
None train crimson-flower-4019 https://wandb.ai/ai2cm/microphysics-emulation/runs/3r8guma7
ablate-gscond-classifier-v1-online piggy-back dainty-aardvark-4018 https://wandb.ai/ai2cm/microphysics-emulation/runs/2r396c2a
ablate-gscond-classifier-v1-online prognostic_evaluation wise-fire-4017 https://wandb.ai/ai2cm/microphysics-emulation/runs/cdohgwi5
ablate-gscond-classifier-v1-offline piggy-back worldly-microwave-4016 https://wandb.ai/ai2cm/microphysics-emulation/runs/39zsxa9l
ablate-gscond-classifier-v1-online prognostic_run cool-sea-4015 https://wandb.ai/ai2cm/microphysics-emulation/runs/2t03v42t
ablate-gscond-classifier-v1-offline prognostic_run stellar-planet-4014 https://wandb.ai/ai2cm/microphysics-emulation/runs/wc7r70r3
combined-gscond-dense-local-75s-v4-offline piggy-back electric-dream-4013 https://wandb.ai/ai2cm/microphysics-emulation/runs/398a7dxe
combined-gscond-dense-local-75s-v4-online piggy-back amber-spaceship-4012 https://wandb.ai/ai2cm/microphysics-emulation/runs/3659lbqc
combined-gscond-dense-local-75s-v4-online prognostic_evaluation blooming-river-4011 https://wandb.ai/ai2cm/microphysics-emulation/runs/w1j2afu7
combined-gscond-dense-local-75s-v4-online prognostic_run astral-totem-4010 https://wandb.ai/ai2cm/microphysics-emulation/runs/wisxw622
```

Use with jq:
```
$ wandb-query -o json | jq .id | head -n 1
"3r8guma7"
```