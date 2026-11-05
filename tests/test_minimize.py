import numpy as np
import pytest
from scipy.optimize import OptimizeWarning, rosen, rosen_der

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
    assert set(res.history) == {"objective", "grad_norm"}
    assert len(res.history["objective"]) == len(res.history["grad_norm"]) == res.nit


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


def test_invalid_jac_rejected():
    with pytest.raises(ValueError, match="finite differences"):
        minimize(rosen, np.zeros(5), jac="cs")


@pytest.mark.parametrize("jac", [None, "3-point"])
def test_central_differences(jac):
    res = minimize(rosen, np.zeros(10), jac=jac)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(10), atol=1e-5)


def test_forward_differences():
    # forward differences are accurate to about sqrt(eps), so they need a looser gtol
    res = minimize(rosen, np.zeros(10), jac="2-point", gtol=1e-4)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(10), atol=1e-3)


def test_finite_differences_count_evaluations():
    calls = []

    def f(x):
        calls.append(1)
        return rosen(x)

    res = minimize(f, np.zeros(5), jac="2-point", maxiter=20)
    assert res.nfev == len(calls)
    assert res.njev > 0


def test_finite_difference_matches_analytic_on_quadratic():
    f, g, x_star = ill_conditioned_quadratic(n=20)
    res = minimize(f, np.zeros(20), jac="3-point", gtol=1e-5)
    assert res.status == 0
    np.testing.assert_allclose(res.x, x_star, atol=1e-5)


def test_finite_diff_rel_step_option():
    default = minimize(rosen, np.zeros(5), jac="3-point", gtol=1e-4)
    res = minimize(rosen, np.zeros(5), jac="3-point", gtol=1e-4, finite_diff_rel_step=1e-4)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(5), atol=1e-3)
    assert not np.array_equal(res.x, default.x)  # the option changed the gradient estimates


@pytest.mark.parametrize("jac", [None, "2-point", "3-point"])
def test_scipy_custom_method_finite_differences(jac):
    from scipy.optimize import minimize as scipy_minimize
    res = scipy_minimize(rosen, np.zeros(5), jac=jac, method=minimize, tol=1e-4)
    np.testing.assert_allclose(res.x, np.ones(5), atol=1e-4)


def test_scipy_custom_method():
    from scipy.optimize import minimize as scipy_minimize
    res = scipy_minimize(rosen, np.zeros(5), jac=rosen_der, method=minimize)
    assert res.fun == pytest.approx(rosen(res.x))


def test_unknown_option_warns():
    with pytest.warns(OptimizeWarning, match="gtoll"):
        minimize(rosen, np.zeros(5), jac=rosen_der, gtoll=1e-8)


def test_scipy_custom_method_no_warning():
    import warnings

    from scipy.optimize import minimize as scipy_minimize
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scipy_minimize(rosen, np.zeros(5), jac=rosen_der, method=minimize)


def test_bounds_not_supported():
    with pytest.raises(ValueError):
        minimize(rosen, np.zeros(5), jac=rosen_der, bounds=[(0, 1)]*5)


def test_callback():
    xs = []
    res = minimize(rosen, np.zeros(5), jac=rosen_der, callback=xs.append)
    assert len(xs) == res.nit
    np.testing.assert_array_equal(xs[-1], res.x)


def test_callback_intermediate_result_and_stop():
    def callback(intermediate_result):
        if intermediate_result.fun < 1:
            raise StopIteration
    res = minimize(rosen, np.zeros(5), jac=rosen_der, callback=callback)
    assert res.status == 6
    assert res.fun < 1


def test_scipy_custom_method_callback():
    from scipy.optimize import minimize as scipy_minimize
    xs = []
    res = scipy_minimize(rosen, np.zeros(5), jac=rosen_der, method=minimize, callback=xs.append)
    assert len(xs) == res.nit


@pytest.mark.parametrize("f, g", [
    (lambda x: np.nan, lambda x: np.ones(3)),
    (lambda x: 1.0, lambda x: np.full(3, np.nan)),
])
def test_not_finite(f, g):
    res = minimize(f, np.zeros(3), jac=g)
    assert res.status == 5
    assert res.nfev == 1


def test_line_search_backs_off_from_inf():
    # f is inf outside x > 0, minimizer at x = 1
    def f(x):
        return -np.sum(np.log(x)) + np.sum(x) if np.all(x > 0) else np.inf
    res = minimize(f, np.full(3, 0.1), jac=lambda x: 1 - 1/x)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(3), atol=1e-6)


def test_unbounded_does_not_succeed():
    res = minimize(lambda x: np.sum(x), np.zeros(3), jac=lambda x: np.ones(3))
    assert not res.success


def test_jac_reusing_output_array():
    buf = np.zeros(10)

    def jac(x):
        buf[:] = rosen_der(x)
        return buf
    res = minimize(rosen, np.zeros(10), jac=jac)
    assert res.status == 0
    np.testing.assert_allclose(res.x, np.ones(10), atol=1e-5)


def test_fun_returning_array_gives_float():
    res = minimize(lambda x: np.array([np.sum(x**2)]), np.ones(3), jac=lambda x: 2*x)
    assert isinstance(res.fun, float)


def test_scipy_tol_sets_gtol():
    import warnings

    from scipy.optimize import minimize as scipy_minimize
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loose = scipy_minimize(rosen, np.zeros(5), jac=rosen_der, method=minimize, tol=1e-2)
        tight = scipy_minimize(rosen, np.zeros(5), jac=rosen_der, method=minimize, tol=1e-8)
    assert loose.status == 0 and tight.status == 0
    assert np.linalg.norm(loose.jac) <= 1e-2
    assert np.linalg.norm(tight.jac) <= 1e-8
    assert loose.nit < tight.nit


def test_gtol_takes_precedence_over_tol():
    res = minimize(rosen, np.zeros(5), jac=rosen_der, gtol=1e-8, tol=1e-2)
    assert np.linalg.norm(res.jac) <= 1e-8


def test_jac_true():
    calls = []

    def fun_and_grad(x):
        calls.append(1)
        return rosen(x), rosen_der(x)

    res = minimize(fun_and_grad, np.zeros(10), jac=True)
    ref = minimize(rosen, np.zeros(10), jac=rosen_der)
    assert res.status == 0
    np.testing.assert_array_equal(res.x, ref.x)
    assert res.nit == ref.nit
    assert len(calls) == res.nfev  # one call per point, not one for f and one for g


def test_jac_true_with_args():
    def fun_and_grad(x, a):
        return np.sum((x - a)**2), 2*(x - a)

    res = minimize(fun_and_grad, np.zeros(5), args=(3.0,), jac=True)
    assert res.status == 0
    np.testing.assert_allclose(res.x, 3.0*np.ones(5), atol=1e-6)


@pytest.mark.parametrize("n, gtol", [(5, 1e-10), (100, 1e-12)])
def test_tight_gtol_converges(n, gtol):
    # tiny search directions near the optimum used to be rejected as "too small"
    res = minimize(rosen, np.zeros(n), jac=rosen_der, gtol=gtol)
    assert res.status == 0
    assert np.linalg.norm(res.jac) <= gtol


def test_precision_limit_status():
    # gtol below what floating point can reach: report status 7, not a plain failure
    res = minimize(rosen, np.zeros(5), jac=rosen_der, gtol=1e-20)
    assert res.status == 7
    assert not res.success
    assert "Precision limit" in res.message
    np.testing.assert_allclose(res.x, np.ones(5), atol=1e-10)


def test_scaled_objective_hits_precision_limit_quickly():
    res = minimize(lambda x: 1e8*rosen(x), np.zeros(10), jac=lambda x: 1e8*rosen_der(x))
    assert res.status == 7
    assert res.nit < 1000
    np.testing.assert_allclose(res.x, np.ones(10), atol=1e-8)


def test_complex_x0_rejected():
    with pytest.raises(ValueError, match="real"):
        minimize(lambda x: np.sum(x**2), np.ones(3) + 0j, jac=lambda x: 2*x)


def test_x0_must_be_1d():
    with pytest.raises(ValueError):
        minimize(lambda x: np.sum(x**2), np.zeros((2, 2)), jac=lambda x: 2*x)


def bdqrtic(x):
    a = x[:-4]
    q = a**2 + 2*x[1:-3]**2 + 3*x[2:-2]**2 + 4*x[3:-1]**2 + 5*x[-1]**2
    return np.sum((-4*a + 3)**2 + q**2)


def bdqrtic_grad(x):
    a = x[:-4]
    q = a**2 + 2*x[1:-3]**2 + 3*x[2:-2]**2 + 4*x[3:-1]**2 + 5*x[-1]**2
    g = np.zeros_like(x)
    g[:-4] += -8*(-4*a + 3) + 4*q*a
    g[1:-3] += 8*q*x[1:-3]
    g[2:-2] += 12*q*x[2:-2]
    g[3:-1] += 16*q*x[3:-1]
    g[-1] += 20*np.sum(q)*x[-1]
    return g


def test_large_minimum_value_converges():
    # f* is about 4e3, so near the minimizer changes in f are at rounding level. The line
    # search used to spin for hundreds of trials there and then fail with status 4
    res = minimize(bdqrtic, np.ones(1000), jac=bdqrtic_grad)
    assert res.status == 0
    assert res.nfev < 10*res.nit
