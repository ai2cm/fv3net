# from .vsrdata import VSRDataset
# from .vsrdata_new import VSRDataset
from .vsrdata_ensemble import VSRDataset

def load_dataset(data_config):
    
    length = data_config["length"]
    logscale = data_config["logscale"]
    multi = data_config["multi"]
    
    train, val = None, None

    train = VSRDataset('train', length, logscale, multi)
    val = VSRDataset('val', length, logscale, multi)
    
    return train, val