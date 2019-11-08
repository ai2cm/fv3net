from vcm.metrics import r2_score

# evaluate
y = model.predict(test, "sample")
r2s = r2_score(test, y, "sample")
