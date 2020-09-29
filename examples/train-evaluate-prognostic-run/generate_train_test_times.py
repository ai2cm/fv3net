# coding: utf-8
import loaders.mappers
import json
url = "gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free/"
data= loaders.mappers.open_baseline_emulator(url)
times = sorted(data)

train_times = times[::5]
test_times = times[2::5]

# train_times = times[:2]
# test_times = times[2:4]

with open("train.json", "w") as f:
    json.dump(train_times, f)
    
with open("test.json", "w") as f:
    json.dump(test_times, f)
    
    
