import os

import download_data
import joblib
import train

cache = joblib.Memory(location=".cache", verbose=10)

fine_res = "gs://vcm-ml-experiments/2020-05-27-40day-fine-res-coarsening/"
local_fine_res = "/home/noahb/data/dev/2020-11-25-fine-res.zarr"
trained_ml = "out.pkl"

cache.cache(download_data.fine_res_to_zarr)(fine_res, local_fine_res)
cache.cache(train.train)(local_fine_res, trained_ml)
