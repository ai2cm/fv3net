from .vsrdata import VSRDataset

def load_dataset(data_config):
    
    channel = data_config["channel"]
    length = data_config["length"]
    
    train, val = None, None

    train = VSRDataset(channel, 'train', length)
    val = VSRDataset(channel, 'val', length)
    
    return train, val