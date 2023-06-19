from .transformer import Transformer
from .autoencoder import Autoencoder
from .sk_transformer import SkTransformer

from typing import Union

ReloadableTransfomer = Union[Autoencoder, SkTransformer]
