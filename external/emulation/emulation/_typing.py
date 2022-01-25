import numpy as np
from typing import MutableMapping

# expected dimensions of state field directly from call_py_fort
# are [feature, sample] where sample is the flattened x,y dim
FortranState = MutableMapping[str, np.ndarray]
