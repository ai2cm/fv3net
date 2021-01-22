import pytest
from offline_ml_diags.keras_loss_history import _get_epoch_losses


@pytest.mark.parametrize(
    "history, key, result",
    [
        ({"loss": [[1.0, 2.0, 3.0]]}, "loss", [2.0]),
        ({"val_loss": [[1.0, 2.0, 3.0]]}, "val_loss", [3.0]),
        ({"loss": [[1.0, 2.0], [3.0, 4.0]]}, "loss", [1.5, 3.5]),
        ({"val_loss": [[1.0, 2.0], [3.0, 4.0]]}, "val_loss", [2.0, 4.0]),
        ({"loss": [[1.0], [3.0]]}, "loss", [1.0, 3.0]),
        ({"loss": [[1.0, 2.0, 3.0]]}, "val_loss", None),
    ],
)
def test__get_epoch_losses(history, key, result):
    assert _get_epoch_losses(history, key) == result
