from fv3fit.data import SyntheticWaves, SyntheticNoise


def test_waves_has_correct_time_shape():
    config = SyntheticWaves(
        nsamples=10,
        nbatch=2,
        ntime=3,
        nx=4,
        nz=5,
        scalar_names=["var_2d"],
        scale_min=0.5,
        scale_max=1.0,
        period_min=8,
        period_max=16,
        wave_type="sinusoidal",
    )
    tfdataset = config.open_tfdataset(local_download_path=None, variable_names=["time"])
    sample = next(iter(tfdataset))
    assert tuple(sample["time"].shape) == (10, 3)


def test_noise_has_correct_time_shape():
    config = SyntheticNoise(
        nsamples=10,
        nbatch=2,
        ntime=3,
        nx=4,
        nz=5,
        scalar_names=["var_2d"],
        noise_amplitude=1.0,
    )
    tfdataset = config.open_tfdataset(local_download_path=None, variable_names=["time"])
    sample = next(iter(tfdataset))
    assert tuple(sample["time"].shape) == (10, 3)
