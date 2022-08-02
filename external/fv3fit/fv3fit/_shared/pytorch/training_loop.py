import tensorflow as tf
from typing import Iterable, Optional, Sequence, Tuple
import dataclasses
import logging
import numpy as np
import torch
import tensorflow_datasets as tfds

logger = logging.getLogger(__name__)


def evaluate_model(config,model,data_iter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss=config.loss.loss()    #torch.nn.MSELoss()
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(torch.as_tensor(np.squeeze(x)).float().to(device))
            y=torch.as_tensor(np.squeeze(y)).float().to(device)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

@dataclasses.dataclass
class TrainingLoopConfig:
    """
    Attributes:
        epochs: number of times to run through the batches when training
        shuffle_buffer_size: size of buffer to use when shuffling samples
        build_samples: the number of samples to pass to build_model
        savemodelpath: name of the file to save the best weights
    """

    n_epoch: int = 20
    buffer_size: int = 50_000
    build_samples: int = 50_000
    savemodelpath: str = 'weight.pt'

    def fit_loop(config,train_model, train_data, validation, optimizer, get_loss) -> None:
        """
        Args:
            train_model: pytorch model to train
            train_data: training dataset containing samples to be passed to the model
            validation: validation dataset to examien the one time prediction
            optimizer: type of optimizer for the model
            get_loss: Multistep loss function 
        """
        train_data.shuffle(buffer_size=config.training_loop.buffer_size)
        train_data=tfds.as_numpy(train_data)
        validation=tfds.as_numpy(validation)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        min_val_loss = np.inf
        for epoch in range(1, config.training_loop.n_epoch + 1):  # loop over the dataset multiple times
            train_model.train()
            l_sum, n = 0.0, 0
            for x, y in train_data:
                optimizer.zero_grad()
                loss = get_loss(config,train_model=train_model , inputs=torch.as_tensor(np.squeeze(x)).float().to(device), labels=torch.as_tensor(np.squeeze(y)).float().to(device))
                loss.backward()
                y=torch.as_tensor(np.squeeze(y)).float().to(device)
                optimizer.step()
                l_sum += loss.item() * y.shape[0]
                n += y.shape[0]
            val_loss = evaluate_model(config,train_model, validation)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(train_model.state_dict(), config.training_loop.savemodelpath)
            # print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)