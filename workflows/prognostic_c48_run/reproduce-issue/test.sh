#!/bin/bash

# using restarts from within the first few segments of the prognostic run results in
# rundir with both atmos_dt_atmos and sfc_dt_atmos netcdfs saved
python3 test.py "20200119"

# using restarts after the sfc_dt_atmos data stopped being saved results in rundir
# with only atmos_dt_atmos.tile*.nc saved but no sfc_dt_atmos
python3 test.py "20200816"