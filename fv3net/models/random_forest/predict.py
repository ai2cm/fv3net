from fv3net.metrics import r2_score
from fv3net.data.calc import mass_integrate

# evaluate
y = model.predict(test, "sample")
r2s = r2_score(test, y, "sample")
