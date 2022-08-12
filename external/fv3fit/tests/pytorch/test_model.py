from fv3fit.pytorch import PytorchModel
from torch import nn
import fv3fit
import numpy as np
import torch


def same_state(model1: nn.Module, model2: nn.Module) -> bool:
    """
    Check if two models have the same state.
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    if state1.keys() != state2.keys():
        return False
    for key in state1.keys():
        if not torch.equal(state1[key], state2[key]):
            return False
    return True


def test_pytorch_model_dump_load(tmpdir):
    scaler = fv3fit.StandardScaler()
    n_features = 2
    n_samples = 10
    state_variables = ["u"]
    data = np.random.uniform(low=-1, high=1, size=(n_samples, n_features))
    scaler.fit(data)
    model = PytorchModel(
        state_variables=state_variables,
        model=nn.Linear(n_features, n_features),
        scalers={"u": scaler},
    )
    model.dump(str(tmpdir))
    reloaded_model = PytorchModel.load(str(tmpdir))
    assert model.state_variables == reloaded_model.state_variables
    assert same_state(model.model, reloaded_model.model)
    assert model.scalers == reloaded_model.scalers
