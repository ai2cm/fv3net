from .vsrdata import VSRDataset

def load_dataset(data_config):
    
    channels = data_config["channels"]
    length = data_config["length"]
    
    train, val = None, None

    train = VSRDataset(channels, 'train', length)
    val = VSRDataset(channels, 'val', length)
    
    return train, val