from fv3net.regression.sklearn import train


def test_train_save_model_succeeds(tmpdir):
    model = object()
    url = str(tmpdir)
    filename = "filename.pkl"
    train.save_model(url, model, filename)
