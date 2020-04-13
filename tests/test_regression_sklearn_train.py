from fv3net.regression.sklearn import train
from datetime import datetime


def test_train_save_output_succeeds(tmpdir):
    model = object()
    config = {"a": 1}
    url = str(tmpdir)
    timesteps = [datetime(2016, 8, 1), datetime(2016, 8, 2)]
    train.save_output(url, model, config, timesteps)
