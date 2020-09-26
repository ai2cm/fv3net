# coding: utf-8
import loaders.mappers
import json
url = "gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free/"
data= loaders.mappers.open_baseline_emulator(url)
times = sorted(data)
with open("train.json", "w") as f:
    json.dump(times[::5], f)
    
with open("test.json", "w") as f:
    json.dump(times[2::5], f)
    
    
