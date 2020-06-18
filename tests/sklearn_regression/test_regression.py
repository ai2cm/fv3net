from loaders import MockDatasetMapper


def test_sklearn_regression():
    """
    python -m fv3net.regression.sklearn \
        $TRAINING_DATA \
        train_sklearn_model_fineres_source.yml  \
        $OUTPUT \
        --no-train-subdir-append
    """
    assert False
