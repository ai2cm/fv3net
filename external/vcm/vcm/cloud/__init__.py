from .fsspec import get_fs, get_protocol, copy
import os

__all__ = [item for item in dir() if not item.startswith("_")]

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    from .gsutil import authenticate

    authenticate(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
