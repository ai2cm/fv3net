import prepare_config


def test_prepare_config_regression(regtest):

    example_config = {
        "base_version": "v0.4",
        "namelist": {
            "coupler_nml": {
                "days": 10,
                "hours": 0,
                "minutes": 0,
                "seconds": 0,
                "dt_atmos": 900,
                "dt_ocean": 900,
                "restart_secs": 0,
            },
            "atmos_model_nml": {"fhout": 0.25},
            "gfs_physics_nml": {"fhzero": 0.25},
            "fv_core_nml": {"n_split": 6},
        },
    }

    output = prepare_config.prepare_config(
        example_config,
        ic_timestep="20160805.000000",
        initial_condition_url="gs://ic-bucket",
        diagnostic_ml=False,
        model_url="gs://ml-model",
        nudge_to_observations=True,
    )

    print(output, file=regtest)
