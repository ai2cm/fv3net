SCALAR_METRIC_KWARGS = {
    "column_integrated_dQ1":
        {"rmse":
            {
                "weights_variables": ["area_weights",],
                "mean_dims": None,
            },
         "bias":
             {
                "weights_variables": ["area_weights"],
                "mean_dims": None,
            },
        },
    "column_integrated_dQ2":
        {"rmse":
            {
                "weights_variables": ["area_weights",],
                "mean_dims": None,
            },
         "bias":
             {
                "weights_variables": ["area_weights",],
                "mean_dims": None,
            },
        },
    "column_integrated_Q1":
        {"rmse":
            {
                "weights_variables": ["area_weights",],
                "mean_dims": None,
            },
         "bias":
             {
                "weights_variables": ["area_weights",],
                "mean_dims": None,
            },
        },
    "column_integrated_Q2":
        {"rmse":
            {
                "weights_variables": ["area_weights"],
                "mean_dims": None,
            },
         "bias":
             {
                "weights_variables": ["area_weights",],
                "mean_dims": None,
            },
        },
    "dQ1":
         {
             "rmse":
                {
                    "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
         },
    "dQ2":
         {
             "rmse":
                {
                    "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
            },
    "Q1":
         {
             "rmse":
                {
                    "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
            },
    "Q2":
         {
             "rmse":
                {
                    "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
             "bias":
                 {
                   "weights_variables": ["delp_weights", "area_weights"],
                    "mean_dims": None,
                },
            }
}
