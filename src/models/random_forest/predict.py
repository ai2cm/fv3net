from src.metrics import r2_score
from src.data.calc import mass_integrate

# evaluate
y = model.predict(test, "sample")
r2s = r2_score(test, y, "sample")
