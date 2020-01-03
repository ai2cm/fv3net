# coding: utf-8
import matplotlib.pyplot as plt
import xarray as xr
from fv3net.regression.sklearn.wrapper import SklearnWrapper
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from vcm.calc.metrics import r2_score

batch = xr.open_dataset("local_batch.nc")

params = {"max_depth": 5, "n_estimators": 20, "verbose": 3, "n_jobs": 10}


# try:
# wrapped_rf = joblib.load("model.pkl")
# except FileNotFoundError:
# rf = RandomForestRegressor(**params)
rf = Ridge(alpha=1e6)
regressor = make_pipeline(StandardScaler(), rf)
# regressor = MLPRegressor(max_iter=200, verbose=True, alpha=10.0)
# regressor = RandomForestRegressor(**params)

scaler = StandardScaler()
model = TransformedTargetRegressor(transformer=scaler, regressor=regressor)
wrapped_rf = SklearnWrapper(model)
input_vars = ["sphum", "T"]
wrapped_rf.fit(
    input_vars=input_vars,
    data=batch.isel(sample=slice(0, 10000)),
    output_vars=["Q1", "Q2"],
    sample_dim="sample",
)
joblib.dump(wrapped_rf, "model.pkl")


# evaluate the model

train = batch.isel(sample=slice(0, 5000))
test = batch.isel(sample=slice(-5000, None))

pred = wrapped_rf.predict(test, sample_dim="sample")
grid = xr.open_dataset(
    "/home/noahb/workspace/explore/noahb/dycoreOnly/atmos_dt_atmos.tile1.nc"
)
r2_q2 = r2_score(test["Q2"], pred["Q2"], sample_dim="sample")

r2_q2 = r2_q2.assign_coords(pfull=grid.pfull)


r2_q2.plot()
plt.show()


def print_r2(test):
    pred = wrapped_rf.predict(test, sample_dim="sample")

    for variable in ["Q1", "Q2"]:
        r2 = r2_score(test[variable], pred[variable], sample_dim="sample")
        print(variable, r2.values.tolist())
        print(variable, float(r2.mean()))


print("Training R2")
print_r2(train)
print()
print("Testing R2")
print_r2(test)
print()
