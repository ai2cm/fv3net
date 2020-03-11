# residuals that the ML is training on
# high resolution tendency - coarse res model's one step tendency
VAR_Q_HEATING_ML = "dQ1"
VAR_Q_MOISTENING_ML = "dQ2"
VAR_Q_U_WIND_ML = "dQU"
VAR_Q_V_WIND_ML = "dQV"

# suffixes denote whether diagnostic variable is from the coarsened
# high resolution prognostic run or the coarse res one step train data run
SUFFIX_HIRES_DIAG = "prog"
SUFFIX_COARSE_TRAIN_DIAG = "train"

DIAG_VARS = [
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "DSWRFtoa",
    "DSWRFsfc",
    "USWRFtoa",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFtoa",
    "ULWRFsfc",
]
RENAMED_PROG_DIAG_VARS = {f"{var}_coarse": f"{var}_prog" for var in DIAG_VARS}
RENAMED_TRAIN_DIAG_VARS = {var: f"{var}_train" for var in DIAG_VARS}


RESTART_VARS = [
    "sphum",
    "T",
    "delp",
    "u",
    "v",
    "slmsk",
    "phis",
    "tsea",
    "slope",
    "DZ",
    "W",
]
