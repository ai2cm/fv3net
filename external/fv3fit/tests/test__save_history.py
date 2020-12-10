import pytest
from fv3fit.keras._save_history import _get_epoch_losses


@pytest.mark.parametrize(
    "history, key, result",
    [
        ({"loss": [[1., 2., 3.]]}, "loss", [2.]),
        ({"val_loss": [[1., 2., 3.]]}, "val_loss", [3.]),
        ({"loss": [[1., 2.], [3., 4.]]}, "loss", [1.5, 3.5]),
        ({"val_loss": [[1., 2.], [3., 4.]]}, "val_loss", [2., 4.]),
        ({"loss": [[1.], [3.]]}, "loss", [1., 3.]),
        ({"loss": [[1., 2., 3.]]}, "val_loss", None),
    ]
)
def test__get_epoch_losses(history, key, result):
    assert _get_epoch_losses(history, key) == result
