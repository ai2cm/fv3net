from fv3net.regression.sklearn import train


def test_train_save_output_succeeds(tmpdir):
    model = object()
    config = {"a": 1}
    url = str(tmpdir)
    timesteps = ['20160801.001500', '20160801.003000']
    train.save_output(url, model, config, timesteps)
