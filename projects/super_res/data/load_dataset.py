from .vsrdata import VSRDataset

def load_dataset(data_config):
    
    channels = data_config["channels"]
    length = data_config["length"]
    logscale = data_config["logscale"]
    quick = data_config["quick"]
    
    train, val = None, None

    train = VSRDataset(channels, 'train', length, logscale, quick)
    val = VSRDataset(channels, 'val', length, logscale, quick)
    
    return train, val