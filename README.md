# Hybrid Cubic CG

A Python implementation of the **Hybrid Cubic Regularization of Conjugate Gradient Minimization Methods (CGM)** for smooth unconstrained optimization.

In our paper, ["Regularized Step Directions in Nonlinear Conjugate Gradient Methods"](https://link.springer.com/article/10.1007/s12532-024-00265-9), we proposed **CGM with hybrid cubic regularization** to improve the step quality of CGM. Standard CGM restarts its search direction when conjugacy is lost (a Powell restart), which can slow convergence from superlinear to linear. Hybrid Cubic CG instead applies cubic regularization to the search direction, but only on those iterations. The regularized direction has a closed form derived from Shanno's memoryless BFGS view of CG, so the method stays matrix-free with the same order of per-iteration cost as standard CG.

In the paper's experiments on the CUTEst test set, it needed fewer iterations, ran faster, and reset the search direction less often than standard CG.

## Installation

```bash
pip install hybrid-cubic-cg
```

## Usage

```python
import numpy as np
from hybrid_cubic_cg import minimize

def f(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def grad(x):
    g = np.zeros_like(x)
    g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
    g[1:] += 200 * (x[1:] - x[:-1]**2)
    return g

res = minimize(f, x0=np.zeros(100), jac=grad)
print(res.x, res.fun, res.nit)
```

The interface follows `scipy.optimize.minimize`.

## Citation

If you use this package, please cite:

```bibtex
@article{buhler2024regularized,
  title   = {Regularized step directions in nonlinear conjugate gradient methods},
  author  = {Buhler, Cassidy K. and Benson, Hande Y. and Shanno, David F.},
  journal = {Mathematical Programming Computation},
  volume  = {16},
  number  = {4},
  pages   = {629--664},
  year    = {2024},
  doi     = {10.1007/s12532-024-00265-9}
}
```

## License

BSD-3-Clause. 
