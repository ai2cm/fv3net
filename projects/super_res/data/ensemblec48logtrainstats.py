import pickle
import numpy as np
from pathlib import Path

precip_folder = Path('./ensemble_c48_trainstats')
precip_folder.mkdir(exist_ok = True, parents = True)

# load the data
with open('ensemble_c48_trainstats/chl.pkl', 'rb') as f:
    chl = pickle.load(f)

precip = chl['PRATEsfc_coarse']
log_chl = {}
log_chl['PRATEsfc_coarse'] = {}
log_chl['PRATEsfc_coarse']['min'] = np.log(precip['min'] - precip['min'] + 1e-14)
log_chl['PRATEsfc_coarse']['max'] = np.log(precip['max'] - precip['min'] + 1e-14)

# save the chl dictionary as pickle
with open(precip_folder / 'log_chl.pkl', 'wb') as f:
    pickle.dump(log_chl, f)