import numpy as np
import pytest
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


def ill_conditioned_quadratic(n=100, seed=1):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    Q = M.T @ M / n + np.diag(np.logspace(0, 3, n))
    b = rng.standard_normal(n)
    return (lambda x: 0.5*x @ Q @ x - b @ x), (lambda x: Q @ x - b), np.linalg.solve(Q, b)


@pytest.mark.parametrize("n", [2, 10, 50, 100, 200, 500])
def test_rosenbrock_from_zero(n):
    res = minimize(rosen, np.zeros(n), jac=rosen_der)
    assert res.status == 0
    assert res.success
    np.testing.assert_allclose(res.x, np.ones(n), atol=1e-5)


@pytest.mark.parametrize("n", [10, 100])
def test_rosenbrock_from_random_start(n):
    x0 = np.random.default_rng(1).uniform(-2, 2, n)
    res = minimize(rosen, x0, jac=rosen_der)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(n), atol=1e-5)


def test_extended_powell():
    x0 = np.tile([3.0, -1.0, 0.0, 1.0], 25)
    res = minimize(extended_powell, x0, jac=extended_powell_grad)
    assert res.status == 0
    assert res.fun < 1e-8


def test_ill_conditioned_quadratic():
    f, g, x_star = ill_conditioned_quadratic()
    res = minimize(f, np.zeros(x_star.size), jac=g)
    assert res.status == 0
    np.testing.assert_allclose(res.x, x_star, atol=1e-6)


def test_readme_example():
    def f(x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def grad(x):
        g = np.zeros_like(x)
        g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
        g[1:] += 200 * (x[1:] - x[:-1]**2)
        return g

    res = minimize(f, x0=np.zeros(100), jac=grad)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(100), atol=1e-5)


@pytest.mark.parametrize("pertlim", [1, 2, 3, 5])
def test_lambda_update_limit(pertlim):
    res = minimize(rosen, np.zeros(50), jac=rosen_der, pertlim=pertlim)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(50), atol=1e-5)


def test_cubic_regularization_is_used():
    res = minimize(rosen, np.zeros(10), jac=rosen_der)
    assert res.invoked_cubic is True


def test_iteration_limit():
    res = minimize(rosen, np.zeros(10), jac=rosen_der, maxiter=3)
    assert res.status == 2
    assert not res.success
    assert res.nit == 3


def test_result_fields():
    res = minimize(rosen, np.zeros(5), jac=rosen_der)
    for field in ["x", "fun", "jac", "nit", "nfev", "njev", "status",
                  "success", "message", "invoked_cubic", "time", "history"]:
        assert field in res
    assert res.fun == pytest.approx(rosen(res.x))
    np.testing.assert_allclose(res.jac, rosen_der(res.x))
    assert len(res.history["objective"]) == len(res.history["gradNorm"])


def test_x0_not_modified():
    x0 = np.zeros(10)
    minimize(rosen, x0, jac=rosen_der)
    np.testing.assert_array_equal(x0, np.zeros(10))


def test_already_optimal():
    res = minimize(rosen, np.ones(10), jac=rosen_der)
    assert res.status == 0
    assert res.nit == 0



def test_args_passed_to_fun_and_jac():
    def f(x, a):
        return np.sum((x - a)**2)

    def grad(x, a):
        return 2*(x - a)

    res = minimize(f, np.zeros(5), args=(3.0,), jac=grad)
    assert res.status == 0
    np.testing.assert_allclose(res.x, 3.0*np.ones(5), atol=1e-6)


def test_jac_must_be_callable():
    with pytest.raises(ValueError):
        minimize(rosen, np.zeros(5))


def test_scipy_custom_method():
    from scipy.optimize import minimize as scipy_minimize
    res = scipy_minimize(rosen, np.zeros(5), jac=rosen_der, method=minimize)
    assert res.fun == pytest.approx(rosen(res.x))
