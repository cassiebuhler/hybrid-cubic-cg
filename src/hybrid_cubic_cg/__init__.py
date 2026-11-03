from importlib.metadata import PackageNotFoundError, version

from ._minimize import minimize

__all__ = ["minimize"]

try:
    __version__ = version("hybrid-cubic-cg")
except PackageNotFoundError:  # running from source without installing
    __version__ = "unknown"
