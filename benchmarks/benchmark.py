"""
Compare hybrid cubic CG with SciPy's CG and L-BFGS-B.

    python benchmarks/benchmark.py            # built-in test problems
    python benchmarks/benchmark.py --cutest   # also CUTEst problems (needs pycutest)

The stopping tests differ: hybrid-cubic-cg stops when the 2-norm of the gradient is
below gtol, SciPy's CG uses the infinity norm, and L-BFGS-B also stops when f stops
changing. The "status" column is each method's own code (they differ between
methods; for hybrid-cubic-cg see the README). The ||g|| column (2-norm at the final
point) puts them on the same footing.

The CUTEst problems are a subset of those used in the paper. Installing pycutest
requires CUTEst itself; see https://jfowkes.github.io/pycutest/. Set PYCUTEST_CACHE to
keep pycutest's compiled problems out of the working directory.
"""
import argparse
import time

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import rosen, rosen_der

from hybrid_cubic_cg import minimize


def extended_powell(x):
    a, b, c, d = x[0::4], x[1::4], x[2::4], x[3::4]
    return np.sum((a + 10*b)**2 + 5*(c - d)**2 + (b - 2*c)**4 + 10*(a - d)**4)


def extended_powell_grad(x):
    a, b, c, d = x[0::4], x[1::4], x[2::4], x[3::4]
    g = np.zeros_like(x)
    g[0::4] = 2*(a + 10*b) + 40*(a - d)**3
    g[1::4] = 20*(a + 10*b) + 4*(b - 2*c)**3
    g[2::4] = 10*(c - d) - 8*(b - 2*c)**3
    g[3::4] = -10*(c - d) - 40*(a - d)**3
    return g


def builtin_problems():
    for n in (100, 1000, 10000):
        yield f"Rosenbrock n={n}", rosen, rosen_der, np.zeros(n)
    for n in (100, 1000):
        yield f"Rosenbrock random start n={n}", rosen, rosen_der, np.random.default_rng(0).uniform(-2, 2, n)
    for n in (100, 1000):
        yield f"Extended Powell n={n}", extended_powell, extended_powell_grad, np.tile([3.0, -1.0, 0.0, 1.0], n // 4)


# Unconstrained CUTEst problems with variable size: (name, SIF parameters)
CUTEST_PROBLEMS = [
    ("ARWHEAD", {"N": 1000}),
    ("BDQRTIC", {"N": 1000}),
    ("BROYDN7D", {}),  # default N=5000; N=1000 is not an allowed size
    ("CRAGGLVY", {"M": 499}),
    ("DQDRTIC", {"N": 1000}),
    ("EDENSCH", {"N": 2000}),
    ("ENGVAL1", {"N": 1000}),
    ("FLETCHCR", {"N": 1000}),
    ("FREUROTH", {"N": 1000}),
    ("NONDIA", {"N": 1000}),
    ("POWER", {"N": 1000}),
    ("SROSENBR", {}),  # default N=5000; N=1000 is not an allowed size
    ("TRIDIA", {"N": 1000}),
    ("WOODS", {"NS": 1000}),
]


def cutest_problems():
    import pycutest

    for name, params in CUTEST_PROBLEMS:
        try:
            p = pycutest.import_problem(name, sifParams=params or None)
        except Exception as e:
            # the size parameter may not be valid for this SIF file: fall back to the default size
            print(f"{name} with {params} failed ({e}); trying the default size")
            try:
                p = pycutest.import_problem(name)
            except Exception as e:  # problem missing or failed to compile
                print(f"skipping {name}: {e}")
                continue
        yield f"{name} n={p.n}", p.obj, p.grad, p.x0


def run(label, fun, jac, x0):
    rows = []
    for method in ("hybrid-cubic-cg", "CG", "L-BFGS-B"):
        t = time.perf_counter()
        if method == "hybrid-cubic-cg":
            res = minimize(fun, x0, jac=jac, gtol=1e-6)
        else:
            res = scipy_minimize(fun, x0, jac=jac, method=method,
                                 options={"gtol": 1e-6, "maxiter": 10000})
        elapsed = time.perf_counter() - t
        gnorm = np.linalg.norm(jac(res.x))
        rows.append((method, res.nit, res.nfev, gnorm, elapsed, res.success, res.status))
    print(f"\n{label}")
    print(f"  {'method':<16}{'iters':>8}{'nfev':>8}{'||g||':>11}{'time (s)':>10}  ok     status")
    for method, nit, nfev, gnorm, elapsed, ok, status in rows:
        print(f"  {method:<16}{nit:>8}{nfev:>8}{gnorm:>11.1e}{elapsed:>10.3f}  {ok!s:<6} {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutest", action="store_true", help="also run CUTEst problems via pycutest")
    args = parser.parse_args()
    if args.cutest:
        try:
            import pycutest  # noqa: F401
        except ImportError:
            parser.error("--cutest needs pycutest and CUTEst: see https://jfowkes.github.io/pycutest/")

    for problem in builtin_problems():
        run(*problem)
    if args.cutest:
        for problem in cutest_problems():
            run(*problem)


if __name__ == "__main__":
    main()
