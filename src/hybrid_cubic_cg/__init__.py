"""
Nonlinear conjugate gradient with hybrid cubic regularization.

Implements the Hybrid Cubic Regularization of CGM from
Buhler, Benson & Shanno (2024), "Regularized step directions in nonlinear
conjugate gradient methods", Mathematical Programming Computation 16(4).
https://doi.org/10.1007/s12532-024-00265-9

The solver is ``minimize``, which follows the interface of
``scipy.optimize.minimize`` and can also be passed to it as ``method=minimize``.
"""
from importlib.metadata import PackageNotFoundError, version

from ._minimize import minimize

__all__ = ["minimize"]

try:
    __version__ = version("hybrid-cubic-cg")
except PackageNotFoundError:  # running from source without installing
    __version__ = "unknown"
