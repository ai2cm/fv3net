from .load_dataset import load_dataset
from torch.utils.data import DataLoader

def load_data(data_config, batch_size, num_workers = 4, pin_memory = True):
    
    train, val = load_dataset(data_config)

    train = DataLoader(
                        train,
                        batch_size = batch_size,
                        shuffle = True,
                        num_workers = num_workers,
                        pin_memory = pin_memory,
                      )

    val = DataLoader(
                        val,
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = num_workers,
                        pin_memory = pin_memory,
                    )
        
    return train, val