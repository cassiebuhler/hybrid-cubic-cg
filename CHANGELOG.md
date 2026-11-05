# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-05

First release.

- `minimize`: nonlinear conjugate gradient with hybrid cubic regularization
  (Buhler, Benson & Shanno, 2024), with an interface that follows
  `scipy.optimize.minimize` and can be passed to it as `method=minimize`.
- Gradients from a callable, from `fun` itself (`jac=True`), or by finite
  differences: central by default (`jac=None` or `'3-point'`), or forward
  (`'2-point'`).
- Line search accepts a point whose value is within rounding error of the
  current one when the slope condition holds (approximate Wolfe condition), so
  problems with a large optimal value converge instead of failing near the
  minimizer.
- Callbacks in both SciPy styles, with early stopping via `StopIteration`.
- Benchmark script comparing against SciPy's `CG` and `L-BFGS-B`.

[0.1.0]: https://github.com/cassiebuhler/hybrid-cubic-cg/releases/tag/v0.1.0
