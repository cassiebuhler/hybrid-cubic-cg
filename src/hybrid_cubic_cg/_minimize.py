"""
Nonlinear conjugate gradient with hybrid cubic regularization.

Implements the Hybrid Cubic Regularization of CGM from
Buhler, Benson & Shanno (2024), "Regularized step directions in nonlinear
conjugate gradient methods", Mathematical Programming Computation 16(4).
"""
import time

import numpy as np
from scipy import optimize


def minimize(fun, x0, args=(), jac=None, *, maxiter=10000, gtol=1e-6, pertlim=5,
             disp=False, **unused):
    """
    Minimize a smooth function with hybrid cubic regularized conjugate gradient.

    Parameters
    ----------
    fun : callable
        Objective function, ``fun(x, *args) -> float``.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Extra arguments passed to ``fun`` and ``jac``.
    jac : callable
        Gradient of ``fun``, ``jac(x, *args) -> ndarray`` of shape (n,).
    maxiter : int, optional
        Maximum number of iterations.
    gtol : float, optional
        Converged when ``||jac(x)|| <= gtol``.
    pertlim : int, optional
        Maximum number of lambda values tried for cubic regularization
        (U in Algorithm 2). When reached, a restart is done instead.
    disp : bool, optional
        Print a summary when finished.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        Fields: ``x``, ``fun``, ``jac``, ``nit``, ``nfev``, ``njev``,
        ``status``, ``success``, ``message``, ``invoked_cubic``, ``time``,
        and ``history``.

        Status codes: 0 = solved, 1 = search direction is not a descent
        direction, 2 = iterations limit reached, 3 = search direction is
        undefined, 4 = line search failed.
    """
    # initialize
    t_start = time.perf_counter()

    # Wrap fun and jac so they receive the extra args
    if not callable(jac):
        raise ValueError('jac must be a callable that returns the gradient.')
    _fun, _jac = fun, jac

    def fun(x):
        return _fun(x, *args)

    def jac(x):
        return _jac(x, *args)

    # Counters and outputs
    k = 0
    nfev = 0
    njev = 0
    status = None
    objs = []
    gradNorms = []
    invokedCubic = False

    # Current point and gradient (c is the gradient, following Conmin's notation)
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    c = jac(x)
    njev += 1
    f = fun(x)
    nfev += 1
    alpha = 1.0
    dg_prev = None  # slope g'd from the previous line search, for the initial step guess

    # Restart and regularization control
    stage = 0      # 0: first step (H0 = I), 1: second step (H1, eq. 13), 2: regular iterations
    nrst = n       # steps since the last restart; nrst == n forces a restart
    pertcnt = 0    # lambda values tried in this episode (u in Algorithm 2)
    lam = 0.0      # cubic regularization parameter; 0 when not regularizing

    # Saved at x_k before each line search, so a rejected trial point can be rolled back
    xprev = None
    c0 = None      # gradient at x_k
    c00 = None     # gradient at x_{k-1}
    alpha0 = None
    dx0 = None
    stage0 = None

    # Main loop: Algorithm 2 (Hybrid Cubic Regularization of CGM)
    while k < maxiter:
        k += 1
        cTc = np.dot(c, c)

        # Check for convergence
        if (np.sqrt(cTc) <= gtol):
            status = 0  # solved
            break

        # Compute step direction
        if stage == 0:
            dx = -c
        else:
            # Powell test (eq. 7): if conjugacy is lost, roll back and try a regularized step
            if (nrst != n) and (stage > 1) and (abs(np.dot(c, c0)/cTc) > 0.2) and (pertcnt < pertlim):
                invokedCubic = True
                objs.pop()
                gradNorms.pop()
                prat = abs(np.dot(c, c0)/cTc)
                alpha = alpha0
                stage = stage0
                x = xprev
                f = f0
                dx = dx0
                c = c0
                c0 = c00
                cTc = np.dot(c, c)
                if lam == 0:
                    lam = prat/0.2
                else:
                    lam = 2*lam
                pertcnt += 1
                k -= 1

            else:
                if pertcnt >= pertlim and abs(np.dot(c, c0)/cTc) > 0.2:
                    nrst = n    # U values of lambda tried: accept this point and restart here
                dx0 = dx
                lam = 0
                pertcnt = 0
            # If performing a restart, update the Beale restart vectors
            if nrst == n:
                pt = alpha*dx
                yt = c - c0
                ytTyt = np.dot(yt, yt)
                ptTyt = np.dot(pt, yt)
                ptTpt = np.dot(pt, pt)

            p = alpha*dx
            y = c - c0
            pTc = np.dot(pt, c)
            yTc = np.dot(yt, c)

            u1 = -pTc/ytTyt
            u2 = 2*pTc/ptTyt - yTc/ytTyt
            scale = ptTyt/ytTyt

            if pertcnt == 0:
                dx = -scale*c - u1*yt - u2*pt
            else:
                bracket = lam*lam + 2*lam*ytTyt/ptTyt + ytTyt/ptTpt
                a = -ptTyt/(lam*ptTyt + ytTyt)
                b1 = (-lam*ptTyt - 2*ytTyt)*ytTyt/(ptTyt*ptTpt*bracket*(lam*ptTyt + ytTyt))
                dcoef = lam/(bracket*(lam*ptTyt + ytTyt))
                e = ytTyt/(ptTpt*bracket*(lam*ptTyt + ytTyt))
                dx = a*c + b1*pTc*pt + dcoef*yTc*yt + e*yTc*pt + e*pTc*yt

            if nrst != n:
                if pertcnt == 0:
                    u1 = -np.dot(y, pt)/ytTyt
                    u2 = -np.dot(y, yt)/ytTyt + 2*np.dot(y, pt)/ptTyt
                    pTy = np.dot(p, y)
                    Hty = ptTyt/ytTyt*y + u1*yt + u2*pt
                    yHty = np.dot(Hty, y)

                    u1 = -np.dot(p, c)/pTy
                    u2 = (yHty/pTy + 1)*np.dot(p, c)/pTy - np.dot(c, Hty)/pTy
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

                    u10 = np.dot(Hlam_y, c)
                    u11 = np.dot(ptilde, c)
                    u12 = np.dot(Hlam_y, y)
                    u13 = np.dot(ptilde, y)
                    u14 = np.dot(ptilde, p)
                    u15 = np.dot(p, y)
                    denom = lam*u14*(u15 + u12) + u13*u13
                    if (not np.isfinite(denom)) or denom <= np.finfo(float).eps*(abs(lam*u14*(u15 + u12)) + u13*u13):
                        status = 3
                        break
                    dx = dx + lam*u14*u10*Hlam_y/denom - (u15 + u12)*u11*ptilde/denom + u13*u10*ptilde/denom + u13*u11*Hlam_y/denom

        # Check that the search direction is a descent direction
        dxTc = np.dot(dx, c)
        if dxTc > 0:
            status = 1
            break
        # Save the current point
        xprev = x
        f0 = f
        if c0 is not None:
            c00 = c0
        c0 = c
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

        # Initial step length guess (as in Conmin-CG)
        if stage <= 1:
            alpha = 1.0/np.sqrt(cTc)
        elif nrst == 1:
            alpha = 1.0
        else:
            alpha = alpha*dg_prev/dxTc
        dg_prev = dxTc

        # Line search (Shanno's inexact line search from Conmin-CG)
        alpha, f_new, c_new, ok, ntry = line_search(fun, jac, xprev, f0, dx, dxTc, alpha, gtol)
        nfev += ntry
        njev += ntry

        if not ok:
            if stage <= 1:
                status = 4  # failed even along steepest descent
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
        c = c_new
        objs.append(f)
        gradNorms.append(np.linalg.norm(c))

    elapsedTime = time.perf_counter() - t_start
    fval = f

    if status is None:
        status = 2

    messages = {
        0: 'Optimization terminated successfully.',
        1: 'Search direction is not a descent direction.',
        2: 'Iterations limit reached.',
        3: 'Search direction is undefined.',
        4: 'Line search failed.',
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
        'gradNorm': gradNorms,
        'status': status,
        'time': elapsedTime,
        'iters': len(objs),
        'invokedCubic': invokedCubic,
    }

    return optimize.OptimizeResult(
        x=x,
        fun=fval,
        jac=c,
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
    Shanno's inexact line search, as implemented in Conmin-CG.

    Finds alpha with sufficient decrease, f(x0 + alpha*dx) <= f0 + 1e-4*alpha*dg,
    and a flatter slope, |jac(x0 + alpha*dx)'dx| < 0.9*|dg|, using Davidon's
    cubic interpolation. dg = jac(x0)'dx must be negative.

    Returns
    -------
    alpha : the step length
    f : the function value at the new point
    c : the gradient at the new point
    ok : whether the search succeeded
    ntry : the number of trial points (each costs one fun and one jac evaluation)
    """
    fmin = f0
    ap, fp, dp = 0.0, f0, dg  # previous trial point: step, f, slope
    dxTdx = np.dot(dx, dx)
    ntry = 0
    while ntry < maxtry:
        if alpha*dxTdx < 1e-20:
            return alpha, f0, None, False, ntry  # step has shrunk to nothing

        x = x0 + alpha*dx
        f = fun(x)
        c = jac(x)
        ntry += 1
        dalpha = np.dot(c, dx)  # slope at alpha

        # f went up but the slope is still negative: went too far, retry from alpha/3
        if f > (1.0 + 1e-12)*fmin and dalpha < -1e-12:
            alpha = alpha/3.0
            ap, fp, dp = 0.0, fmin, dg
            continue

        # Accept: sufficient decrease and slope reduced (after at least two trials,
        # unless the slope is already essentially zero)
        if f <= fmin + 1e-4*alpha*dg and abs(dalpha/dg) < 0.9:
            if ntry >= 2 or abs(dalpha/dg) < gtol:
                return alpha, f, c, True, ntry

        # Davidon's formula for the minimum of the cubic through (ap, fp, dp) and (alpha, f, dalpha)
        u1 = dp + dalpha - 3.0*(fp - f)/(ap - alpha)
        u2 = np.sqrt(max(u1*u1 - dp*dalpha, 0.0))
        at = alpha - (alpha - ap)*(dalpha + u2 - u1)/(dalpha - dp + 2.0*u2)

        if dalpha/dp > 0:
            # minimum not bracketed yet: extrapolate (double) or step back (halve)
            if dalpha <= 0 or at <= 0 or at >= 0.99*min(alpha, ap):
                if dalpha > 0 or at <= 1.01*max(ap, alpha):
                    at = 2*max(ap, alpha) if dalpha <= 0 else min(ap, alpha)/2
        else:
            # minimum bracketed: bisect if the cubic's guess is too close to an end
            if at < 1.01*min(alpha, ap) or at > 0.99*max(alpha, ap):
                at = (alpha + ap)/2

        ap, fp, dp = alpha, f, dalpha
        alpha = at

    return alpha, f, c, True, ntry  # out of trials: take the last point
