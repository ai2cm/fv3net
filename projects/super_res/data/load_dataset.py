from .vsrdata import VSRDataset

def load_dataset(data_config):
    
    channels = data_config["channels"]
    length = data_config["length"]
    logscale = data_config["logscale"]
    
    train, val = None, None

    train = VSRDataset(channels, 'train', length, logscale)
    val = VSRDataset(channels, 'val', length, logscale)
    
    return train, val