from typing import Sequence, Tuple
from .. import shared
import xarray as xr
from loaders import batches


# def get_X_y(
#     data_path: str, train_config: shared.ModelTrainingConfig
# ) -> Tuple[Sequence[xr.Dataset], Sequence[xr.Dataset]]:
#     """
#     Args:
#         data_path: data location
#         train_config: model training configuration

#     Returns:
#         Sequence of datasets iterated over in training
#     """
#     batch_function = getattr(batches, train_config.batch_function)
#     X = batch_function(
#         data_path,
#         list(train_config.input_variables),
#         **train_config.batch_kwargs,
#     )
#     y = batch_function(
#         data_path,
#         list(train_config.output_variables),
#         **train_config.batch_kwargs,
#     )
#     return X, y
