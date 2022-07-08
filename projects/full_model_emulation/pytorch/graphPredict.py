class PytorchModel(Predictor):
    pass

    # def predict(self, X: xr.Dataset) -> xr.Dataset:
    #     """Predict an output xarray dataset from an input xarray dataset."""
    #     X = X.transpose(["time", "tile", "x", "y", "z"])
    #     inputs = [X_stacked[name].values for name in self.input_variables]
    #     outputs = self.model.predict(inputs)
    #     if isinstance(outputs, np.ndarray):
    #         outputs = [outputs]
    #     # turn outputs into an xarray dataset
    #     return match_prediction_to_input_coords(X, return_ds)
