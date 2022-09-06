import sys

sys.path.append("../pytorch")

import graph

@pytest.mark.slow


def test_input_output_size_correct(model_type: str, regtest):
    """
    """
    sample_func = get_data()
    assert_correct_output(model_type, sample_func)


@pytest.mark.slow
def test_train_graph(model_type: str):
    """
    """
    sample_func = get_data()
    assert_graph_training(model_type, sample_func)

# try running the training
pass
