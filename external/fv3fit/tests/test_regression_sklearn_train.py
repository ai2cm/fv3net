from fv3fit.sklearn._train import save_model


def test_train_save_model_succeeds(tmpdir):
    model = object()
    url = str(tmpdir)
    filename = "filename.pkl"
    save_model(url, model, filename)
