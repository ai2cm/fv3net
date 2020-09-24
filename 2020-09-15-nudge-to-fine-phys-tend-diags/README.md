# 2020-09-15 Hybrid Experiments

This directory contains configurations to run a number of experiments related to
the hybrid ML approach.

## New training simulations

1. To run a physics-on nudged-to-fine simulation with physics tendency component
   fortran diagnostics output every two hours use:
```
$ make submit_physics_on_nudge_to_fine_training_simulation
```
2. To run a clouds-off nudged-to-fine simulation with physics tendency component
   fortran diagnostics output every two hours use:
```
$ make submit_clouds_off_nudge_to_fine_training_simulation
```

## New ML experiments

1. To train an RF, run a prognostic run, and compute diagnostics for a hybrid
   run with pQs computed using the fortran diagnostics from training simulation
   (1) use:
```
$ make submit_train_diags_prog_hybrid_fortran_pQ_no_nudging_tendencies
```
2. To train an RF, run a prognostic run, and compute diagnostics for a hybrid
   run with pQs computed using the fortran diagnostics from training simulation
   (1) and nudging tendencies from the nudged-to-fine simulation added to the
   dQs use:
```
$ make submit_train_diags_prog_hybrid_fortran_pQ_nudge_to_fine_tendencies
```
3. To train an RF, run a prognostic run, and compute diagnostics for a hybrid
   run with pQs computed using the fortran diagnostics from training simulation
   (1) and the temperature nudging tendency from the X-SHiELD simulation added
   to the dQs use:
```
$ make submit_train_diags_prog_hybrid_fortran_pQ_xshield_nudging_tendency
```
4. To train an RF, run a prognostic run, and compute diagnostics for a hybrid
   run with pQs computed using the wrapper physics tendencies from training
   simulation (2) use:
```
$ make submit_train_diags_prog_hybrid_nudge_to_fine_pQ_no_nudging_tendencies
```
