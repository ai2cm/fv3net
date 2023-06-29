from .autoencoder import Autoencoder
from .sk_transformer import SkTransformer
from .transformer import encode_columns
from typing import Union

ReloadableTransfomer = Union[Autoencoder, SkTransformer]
