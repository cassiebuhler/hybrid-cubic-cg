# Benchmarks

Compares hybrid cubic CG with SciPy's `CG` and `L-BFGS-B`.

```bash
python benchmarks/benchmark.py            # built-in problems
python benchmarks/benchmark.py --cutest   # also CUTEst problems (requires pycutest)
```

## Results

`gtol=1e-6`, at most 10,000 iterations. Each cell is iterations / function evaluations (final gradient norm). ✗ means the method did not converge.

| Problem | n | Hybrid cubic CG | SciPy CG | SciPy L-BFGS-B |
|---|--:|--:|--:|--:|
| Rosenbrock | 100 | 666 / 1,872 (7.8e-7) | 1,202 / 1,844 (2.0e-6) | 499 / 595 (6.6e-4) |
| Rosenbrock | 1,000 | 5,366 / 13,423 (7.9e-7) | 10,000 / 15,101 (6.2e0) ✗ | 4,915 / 5,769 (7.8e-4) |
| Rosenbrock | 10,000 | 10,000 / 26,462 (1.2e1) ✗ | 10,000 / 15,146 (6.4e0) ✗ | 10,000 / 11,726 (1.8e1) ✗ |
| Rosenbrock random start | 100 | 685 / 2,000 (7.8e-7) | 1,316 / 2,007 (3.6e-6) | 515 / 622 (8.8e-4) |
| Rosenbrock random start | 1,000 | 5,241 / 13,443 (7.3e-7) | 10,000 / 14,938 (9.0e0) ✗ | 3,673 / 4,275 (9.1e-4) |
| Extended Powell | 100 | 62 / 395 (1.0e-6) | 76 / 156 (4.3e-6) | 43 / 45 (3.0e-5) |
| Extended Powell | 1,000 | 93 / 501 (6.5e-7) | 56 / 116 (1.8e-5) | 39 / 43 (4.8e-6) |
| ARWHEAD | 1,000 | 14 / 107 (1.2e-9) | 5 / 50 (6.5e-6) ✗ | 12 / 13 (1.5e-5) |
| BDQRTIC | 1,000 | 597 / 6,302 (9.3e-7) | 154 / 444 (1.5e-4) ✗ | 68 / 77 (1.3e-1) |
| BROYDN7D | 5,000 | 1,500 / 3,682 (9.3e-7) | 1,685 / 2,393 (7.0e-6) ✗ | 1,632 / 1,676 (2.0e-2) |
| CRAGGLVY | 1,000 | 116 / 528 (8.4e-7) | 100 / 273 (1.2e-5) ✗ | 54 / 59 (9.2e-3) |
| DQDRTIC | 1,000 | 9 / 29 (4.1e-9) | 6 / 15 (2.4e-10) | 12 / 18 (3.2e-5) |
| EDENSCH | 2,000 | 37 / 180 (8.8e-7) | 26 / 126 (4.7e-6) ✗ | 22 / 27 (1.3e-2) |
| ENGVAL1 | 1,000 | 24 / 106 (7.8e-7) | 20 / 99 (2.1e-6) ✗ | 15 / 17 (7.6e-4) |
| FLETCHCR | 1,000 | 5,656 / 14,677 (9.3e-7) | 10,000 / 15,012 (1.4e1) ✗ | 4,915 / 5,771 (4.2e-4) |
| FREUROTH | 1,000 | 106 / 1,039 (9.9e-7) | 26 / 115 (3.6e-4) ✗ | 24 / 33 (1.4e-1) |
| NONDIA | 1,000 | 13 / 69 (1.5e-7) | 11 / 33 (2.2e-12) | 18 / 21 (8.3e-5) |
| POWER | 1,000 | 248 / 1,108 (9.8e-7) | 165 / 281 (3.6e-6) | 128 / 131 (1.2e-5) |
| SROSENBR | 5,000 | 45 / 356 (7.7e-7) | 44 / 98 (2.0e-8) | 47 / 60 (2.4e-5) |
| TRIDIA | 1,000 | 726 / 1,604 (8.4e-7) | 1,754 / 2,633 (4.3e-6) | 435 / 447 (1.1e-2) |
| WOODS | 4,000 | 40 / 218 (7.7e-8) | 61 / 126 (4.0e-5) | 89 / 119 (1.9e-6) |

Notes:

- L-BFGS-B also stops when `f` stops decreasing, so it often finishes with a larger gradient.
- Hybrid cubic CG's line search takes at least two trial points per iteration, so it uses more function evaluations per iteration than SciPy's `CG`.
