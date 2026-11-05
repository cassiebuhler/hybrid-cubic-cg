"""
Nonlinear conjugate gradient with hybrid cubic regularization.

Implements the Hybrid Cubic Regularization of CGM from
Buhler, Benson & Shanno (2024), "Regularized step directions in nonlinear
conjugate gradient methods", Mathematical Programming Computation 16(4).
"""
from __future__ import annotations

import inspect
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

EPS = np.finfo(float).eps


def minimize(
    fun: Callable[..., Any],
    x0: ArrayLike,
    args: tuple = (),
    jac: Callable[..., ArrayLike] | bool | str | None = None,
    *,
    maxiter: int = 10000,
    gtol: float | None = None,
    tol: float | None = None,
    pertlim: int = 5,
    finite_diff_rel_step: float | None = None,
    disp: bool = False,
    callback: Callable[..., Any] | None = None,
    bounds: Any = None,
    constraints: Any = (),
    **unused: Any,
) -> optimize.OptimizeResult:
    """
    Minimize a smooth function with hybrid cubic regularized conjugate gradient.

    Parameters
    ----------
    fun : callable
        Objective function, ``fun(x, *args) -> float``. If ``jac=True``,
        ``fun`` returns both the value and the gradient: ``(f, g)``.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Extra arguments passed to ``fun`` and ``jac``.
    jac : callable, True, None, '2-point' or '3-point', optional
        Gradient of ``fun``, ``jac(x, *args) -> ndarray`` of shape (n,).
        If ``True``, ``fun`` returns the gradient along with the value, and
        it is called once per point. If ``None`` (the default) or
        ``'2-point'``, the gradient is estimated by forward differences, which
        costs ``n`` extra evaluations of ``fun`` per gradient; ``'3-point'``
        uses central differences (``2n`` evaluations, more accurate). An
        analytic gradient is much faster and more accurate for large ``n``.
    maxiter : int, optional
        Maximum number of iterations.
    gtol : float, optional
        Converged when ``||jac(x)|| <= gtol``. Default ``1e-6``.
    tol : float, optional
        Same as ``gtol``; this is the name ``scipy.optimize.minimize`` uses.
        If both are given, ``gtol`` is used.
    pertlim : int, optional
        Maximum number of lambda values tried for cubic regularization
        (U in Algorithm 2). When reached, a restart is done instead.
    finite_diff_rel_step : float, optional
        Relative step size for finite differences; the step for ``x[i]`` is
        ``finite_diff_rel_step * max(1, |x[i]|)``. Defaults to ``sqrt(eps)``
        for ``'2-point'`` and ``eps**(1/3)`` for ``'3-point'``.
    disp : bool, optional
        Print a summary when finished.
    callback : callable, optional
        Called after each iteration as ``callback(xk)``, or as
        ``callback(intermediate_result=res)`` with ``res.x`` and ``res.fun``.
        Raise ``StopIteration`` inside it to stop early.
    bounds, constraints
        Not supported; accepted only so the solver can be passed to
        ``scipy.optimize.minimize``.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        Fields: ``x``, ``fun``, ``jac``, ``nit``, ``nfev``, ``njev``,
        ``status``, ``success``, ``message``, ``invoked_cubic``, ``time``,
        and ``history``. ``nfev`` counts every evaluation of ``fun``,
        including those used for finite differences. ``history`` holds
        per-iteration lists ``objective`` and ``grad_norm``.

        Status codes: 0 = solved, 1 = search direction is not a descent
        direction, 2 = iterations limit reached, 3 = search direction is
        undefined, 4 = line search failed, 5 = function or gradient is not
        finite, 6 = stopped by callback, 7 = precision limit reached (the
        step is too small to change x in floating point, usually because x
        is already very close to a minimizer; check ``res.jac``).
    """
    # initialize
    t_start = time.perf_counter()

    # Check options (scipy.optimize.minimize also passes hess=None and hessp=None)
    if bounds is not None or constraints:
        raise ValueError('bounds and constraints are not supported.')
    unused = [key for key, val in unused.items() if val is not None or key not in ('hess', 'hessp')]
    if unused:
        warnings.warn('Unknown solver options: %s' % ', '.join(unused), optimize.OptimizeWarning, stacklevel=2)
    newCallback = callback is not None and set(inspect.signature(callback).parameters) == {'intermediate_result'}

    def report(x, f):
        """Call the callback at an accepted point. Returns True if it raised StopIteration."""
        if callback is None:
            return False
        try:
            if newCallback:
                callback(intermediate_result=optimize.OptimizeResult(x=np.copy(x), fun=f))
            else:
                callback(np.copy(x))
        except StopIteration:
            return True
        return False

    # Convergence tolerance: gtol, else scipy's generic tol, else the default
    if gtol is None:
        gtol = tol if tol is not None else 1e-6

    # Wrap fun and jac so they receive the extra args and count evaluations. The gradient
    # is copied in case jac reuses its output array, since we keep previous gradients
    counts = {'f': 0, 'g': 0}
    last = {'x': None}  # the most recent evaluation, so the same point is never evaluated twice
    if jac is True:
        # fun returns (f, g): evaluate it once per point and keep both results
        _fun_and_jac = fun

        def evaluate(x):
            if last['x'] is None or not np.array_equal(last['x'], x):
                f_val, g_val = _fun_and_jac(x, *args)
                counts['f'] += 1
                counts['g'] += 1
                last['f'] = np.asarray(f_val, dtype=float).item()
                last['g'] = np.array(g_val, dtype=float)
                last['x'] = np.copy(x)
            return last

        def fun(x):
            return evaluate(x)['f']

        def jac(x):
            return np.copy(evaluate(x)['g'])
    else:
        _fun = fun

        def call_fun(x):
            counts['f'] += 1
            return np.asarray(_fun(x, *args), dtype=float).item()

        def fun(x):
            if last['x'] is None or not np.array_equal(last['x'], x):
                last['f'] = call_fun(x)
                last['x'] = np.copy(x)
            return last['f']

        if callable(jac):
            _jac = jac

            def jac(x):
                counts['g'] += 1
                return np.array(_jac(x, *args), dtype=float)
        elif jac is None or jac is False or jac in ('2-point', '3-point'):
            # Finite differences: n extra evaluations of fun per gradient (2n for '3-point')
            central = jac == '3-point'
            if finite_diff_rel_step is None:
                finite_diff_rel_step = EPS**(1/3) if central else EPS**0.5

            def jac(x):
                counts['g'] += 1
                h = finite_diff_rel_step*np.where(x >= 0, 1.0, -1.0)*np.maximum(1.0, np.abs(x))
                h = (x + h) - x  # make the step exactly representable
                f_x = None if central else fun(x)
                g = np.empty_like(x)
                for i in range(x.size):
                    xt = np.copy(x)
                    xt[i] = x[i] + h[i]
                    f_plus = call_fun(xt)
                    if central:
                        xt[i] = x[i] - h[i]
                        g[i] = (f_plus - call_fun(xt))/(2*h[i])
                    else:
                        g[i] = (f_plus - f_x)/h[i]
                return g
        else:
            raise ValueError("jac must be a callable that returns the gradient, True if fun "
                             "returns (value, gradient), or None, '2-point' or '3-point' to "
                             "use finite differences.")

    # Counters and outputs
    k = 0
    status = None
    objs = []
    gradNorms = []
    invokedCubic = False
    newPoint = False  # x is a new point that hasn't been passed to the callback yet

    # Current point and gradient (g is the gradient)
    if np.iscomplexobj(x0):
        raise ValueError('x0 must be real; complex values are not supported.')
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    if x.ndim != 1:
        raise ValueError('x0 must be one-dimensional.')
    n = x.size
    g = jac(x)
    f = fun(x)
    alpha = 1.0
    dg_prev = None  # slope g'd from the previous line search, for the initial step guess

    # Restart and regularization control
    stage = 0      # 0: first step (H0 = I), 1: second step (H1, eq. 13), 2: regular iterations
    nrst = n       # steps since the last restart; nrst == n forces a restart
    pertcnt = 0    # lambda values tried in this episode (u in Algorithm 2)
    lam = 0.0      # cubic regularization parameter; 0 when not regularizing

    # Saved at x_k before each line search, so a rejected trial point can be rolled back
    xprev = None
    f0 = None      # f at x_k
    g0 = None      # gradient at x_k
    g00 = None     # gradient at x_{k-1}
    alpha0 = None
    dx0 = None
    stage0 = None

    # Main loop: Algorithm 2 (Hybrid Cubic Regularization of CGM)
    while k < maxiter:
        k += 1
        gTg = np.dot(g, g)

        # Check that the function and gradient are finite
        if not (np.isfinite(f) and np.isfinite(gTg)):
            status = 5
            break

        # Check for convergence
        if (np.sqrt(gTg) <= gtol):
            status = 0  # solved
            break

        # Compute step direction
        if stage == 0:
            dx = -g
        else:
            # Powell test (eq. 7): if conjugacy is lost, roll back and try a regularized step
            if (nrst != n) and (stage > 1) and (abs(np.dot(g, g0)/gTg) > 0.2) and (pertcnt < pertlim):
                invokedCubic = True
                newPoint = False
                objs.pop()
                gradNorms.pop()
                prat = abs(np.dot(g, g0)/gTg)
                alpha = alpha0
                stage = stage0
                x = xprev
                f = f0
                dx = dx0
                g = g0
                g0 = g00
                gTg = np.dot(g, g)
                if lam == 0:
                    lam = prat/0.2
                else:
                    lam = 2*lam
                pertcnt += 1
                k -= 1

            else:
                # The point is accepted: call the callback, which can raise StopIteration to stop
                newPoint = False
                if report(x, f):
                    status = 6
                    break
                if pertcnt >= pertlim and abs(np.dot(g, g0)/gTg) > 0.2:
                    nrst = n    # U values of lambda tried: accept this point and restart here
                dx0 = dx
                lam = 0
                pertcnt = 0
            # If performing a restart, update the Beale restart vectors
            if nrst == n:
                pt = alpha*dx
                yt = g - g0
                ytTyt = np.dot(yt, yt)
                ptTyt = np.dot(pt, yt)
                ptTpt = np.dot(pt, pt)

            p = alpha*dx
            y = g - g0
            pTg = np.dot(pt, g)
            yTg = np.dot(yt, g)

            u1 = -pTg/ytTyt
            u2 = 2*pTg/ptTyt - yTg/ytTyt
            scale = ptTyt/ytTyt

            if pertcnt == 0:
                dx = -scale*g - u1*yt - u2*pt
            else:
                bracket = lam*lam + 2*lam*ytTyt/ptTyt + ytTyt/ptTpt
                a = -ptTyt/(lam*ptTyt + ytTyt)
                b1 = (-lam*ptTyt - 2*ytTyt)*ytTyt/(ptTyt*ptTpt*bracket*(lam*ptTyt + ytTyt))
                dcoef = lam/(bracket*(lam*ptTyt + ytTyt))
                e = ytTyt/(ptTpt*bracket*(lam*ptTyt + ytTyt))
                dx = a*g + b1*pTg*pt + dcoef*yTg*yt + e*yTg*pt + e*pTg*yt

            if nrst != n:
                if pertcnt == 0:
                    u1 = -np.dot(y, pt)/ytTyt
                    u2 = -np.dot(y, yt)/ytTyt + 2*np.dot(y, pt)/ptTyt
                    pTy = np.dot(p, y)
                    Hty = ptTyt/ytTyt*y + u1*yt + u2*pt
                    yHty = np.dot(Hty, y)

                    u1 = -np.dot(p, g)/pTy
                    u2 = (yHty/pTy + 1)*np.dot(p, g)/pTy - np.dot(g, Hty)/pTy
                    dx = dx - u1*Hty - u2*p
                else:
                    ptTy = np.dot(pt, y)
                    ytTy = np.dot(yt, y)
                    Hlam_y = -(a*y + b1*ptTy*pt + dcoef*ytTy*yt + e*ytTy*pt + e*ptTy*yt)

                    a2 = 1.0/(lam*scale+1.0)
                    b2 = lam*b1
                    d2 = lam*dcoef
                    e2 = lam*e
                    ptTp = np.dot(pt, p)
                    ytTp = np.dot(yt, p)
                    ptilde = a2*p + b2*ptTp*pt + d2*ytTp*yt + e2*ytTp*pt + e2*ptTp*yt

                    u10 = np.dot(Hlam_y, g)
                    u11 = np.dot(ptilde, g)
                    u12 = np.dot(Hlam_y, y)
                    u13 = np.dot(ptilde, y)
                    u14 = np.dot(ptilde, p)
                    u15 = np.dot(p, y)
                    denom = lam*u14*(u15 + u12) + u13*u13
                    if (not np.isfinite(denom)) or denom <= EPS*(abs(lam*u14*(u15 + u12)) + u13*u13):
                        status = 3
                        break
                    dx = (dx + lam*u14*u10*Hlam_y/denom - (u15 + u12)*u11*ptilde/denom
                          + u13*u10*ptilde/denom + u13*u11*Hlam_y/denom)

        # Check that the search direction is a descent direction
        dxTg = np.dot(dx, g)
        if not dxTg < 0:  # also catches dxTg = 0 or NaN
            status = 1
            break
        # Save the current point
        xprev = x
        f0 = f
        if g0 is not None:
            g00 = g0
        g0 = g
        alpha0 = alpha
        stage0 = stage

        if stage == 0:
            stage = 1
        else:
            if pertcnt == 0:
                if nrst == n:
                    nrst = 0
                nrst += 1
                stage = 2

        # Initial step length guess
        if stage <= 1:
            alpha = 1.0/np.sqrt(gTg)
        elif nrst == 1:
            alpha = 1.0
        else:
            alpha = alpha*dg_prev/dxTg
        dg_prev = dxTg

        # Line search
        alpha, f_new, g_new, ls_status = line_search(fun, jac, xprev, f0, dx, dxTg, alpha, gtol)

        if ls_status != 0:
            if stage <= 1:
                status = ls_status  # failed even along steepest descent: 4 or 7
                break
            # restart with steepest descent from the current point
            nrst = n
            stage = 0
            lam = 0.0
            pertcnt = 0
            continue

        # Take the step and update function value and gradient
        x = xprev + alpha*dx
        f = f_new
        g = g_new
        objs.append(f)
        gradNorms.append(np.linalg.norm(g))
        newPoint = True

    # The final point passed no Powell test, so report it here
    if newPoint:
        report(x, f)

    elapsedTime = time.perf_counter() - t_start
    fval = f
    nfev = counts['f']
    njev = counts['g']

    if status is None:
        status = 2

    messages = {
        0: 'Optimization terminated successfully.',
        1: 'Search direction is not a descent direction.',
        2: 'Iterations limit reached.',
        3: 'Search direction is undefined.',
        4: 'Line search failed.',
        5: 'Function or gradient is not finite.',
        6: 'Stopped by callback.',
        7: ('Precision limit reached: the step is too small to change x. '
            'Gradient norm is %.1e; x may already be close enough to optimal.' % np.linalg.norm(g)),
    }

    if disp:
        print(messages[status])
        print('         Current function value: %f' % fval)
        print('         Iterations: %d' % len(objs))
        print('         Function evaluations: %d' % nfev)
        print('         Gradient evaluations: %d' % njev)
        print('         Cubic regularization used: %s' % invokedCubic)

    history = {
        'objective': objs,
        'grad_norm': gradNorms,
    }

    return optimize.OptimizeResult(
        x=x,
        fun=fval,
        jac=g,
        nit=len(objs),
        nfev=nfev,
        njev=njev,
        status=status,
        success=(status == 0),
        message=messages[status],
        invoked_cubic=bool(invokedCubic),
        time=elapsedTime,
        history=history,
    )


def line_search(fun, jac, x0, f0, dx, dg, alpha, gtol, maxtry=300):
    """
    Inexact line search using Davidon's cubic interpolation.

    Finds alpha with sufficient decrease, f(x0 + alpha*dx) <= f0 + 1e-4*alpha*dg,
    and a flatter slope, |jac(x0 + alpha*dx)'dx| < 0.9*|dg|, using Davidon's
    cubic interpolation. dg = jac(x0)'dx must be negative. When the decrease is
    below rounding error in f, f within rounding error of f0 is accepted instead.

    Returns
    -------
    alpha : the step length
    f : the function value at the new point
    g : the gradient at the new point
    status : 0 = success, 4 = failed, 7 = the step is too small to change x
    """
    fmin = f0
    ap, fp, dp = 0.0, f0, dg  # previous trial point: step, f, slope
    eps_x = EPS*np.abs(x0)
    # Allowance for rounding error in f: near a minimizer, f can't resolve the decrease the
    # first test asks for, so accept f up to this much above f0 if the slope test passes
    # (the approximate Wolfe condition of Hager and Zhang). Rounding error in a sum of many
    # terms can be hundreds of eps relative to f, so the margin is well above that
    eps_f = 1e-10*abs(f0)
    ntry = 0
    while ntry < maxtry:
        if np.all(np.abs(alpha*dx) <= eps_x):
            return alpha, f0, None, 7  # step no longer changes x in floating point

        x = x0 + alpha*dx
        f = fun(x)
        g = jac(x)
        ntry += 1
        dalpha = np.dot(g, dx)  # slope at alpha

        # f is not finite (e.g. outside the function's domain), or f went up but the
        # slope is still negative: went too far, retry from alpha/3
        if not (np.isfinite(f) and np.isfinite(dalpha)) or (f > (1.0 + 1e-12)*fmin and dalpha < -1e-12):
            alpha = alpha/3.0
            ap, fp, dp = 0.0, fmin, dg
            continue

        # Accept: sufficient decrease and slope reduced (after at least two trials,
        # unless the slope is already essentially zero)
        if (f <= fmin + 1e-4*alpha*dg or f <= fmin + eps_f) and abs(dalpha/dg) < 0.9:
            if ntry >= 2 or abs(dalpha/dg) < gtol:
                return alpha, f, g, 0

        # Davidon's formula for the minimum of the cubic through (ap, fp, dp) and (alpha, f, dalpha)
        u1 = dp + dalpha - 3.0*(fp - f)/(ap - alpha)
        u2 = np.sqrt(max(u1*u1 - dp*dalpha, 0.0))
        denom = dalpha - dp + 2.0*u2
        at = alpha - (alpha - ap)*(dalpha + u2 - u1)/denom if denom != 0 else np.nan
        bad = not np.isfinite(at)  # the cubic has no usable minimum: use the fallbacks below

        if dalpha*dp > 0:
            # minimum not bracketed yet: extrapolate (double) or step back (halve)
            if bad or dalpha <= 0 or at <= 0 or at >= 0.99*min(alpha, ap):
                if bad or dalpha > 0 or at <= 1.01*max(ap, alpha):
                    at = 2*max(ap, alpha) if dalpha <= 0 else min(ap, alpha)/2
        else:
            # minimum bracketed: bisect if the cubic's guess is too close to an end
            if bad or at < 1.01*min(alpha, ap) or at > 0.99*max(alpha, ap):
                at = (alpha + ap)/2

        ap, fp, dp = alpha, f, dalpha
        alpha = at

    return alpha, f0, None, 4  # out of trials: the last point may not decrease f
