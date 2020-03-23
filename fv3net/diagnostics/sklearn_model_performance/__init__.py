from .data_funcs_sklearn import (
    predict_on_test_data,
    load_high_res_diag_dataset,
    add_column_heating_moistening,
    integrate_for_Q,
    lower_tropospheric_stability,
)

DATASET_NAME_PREDICTION = "prediction"
DATASET_NAME_FV3_TARGET = "C48 target"
DATASET_NAME_SHIELD_HIRES = "coarsened high res"