import .names



def load_test_data(path: str) -> xr.Dataset:
    pass


def load_model(path: str):
    pass


def offline_predction(model, test_dataset):
    """
    does some stack/unstack of loaded test data before giving to wrapper model
    put this here and remove stuff from fv3net.regression.sklearn.test
    """
    pass



def get_var_arrays(ds_test, ds_prediction):
    """ 
    get data arrays for all the necessary quantites to calc metrics
    and diagnostics. 
    net_precip from hires, target, prediction (each is 1 data array)
    net_heating from hires, target, prediction
    dQ1, pQ1, Q1, etc. vertical profiles from 

    return format dict { {var_name}_{source}: DataArray }? then can pass
    to the create_metrics(), create_diagnostics() funcs and the functions
    called within those 
    
    """
    pass


def source_name_from_var(var_name):
    """
    source from var name in format "{var_name}_{source}"
    """
    pass