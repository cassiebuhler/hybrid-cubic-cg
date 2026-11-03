"""
Nonlinear conjugate gradient with hybrid cubic regularization.

Implements the Hybrid Cubic Regularization of CGM from
Buhler, Benson & Shanno (2024), "Regularized step directions in nonlinear
conjugate gradient methods", Mathematical Programming Computation 16(4).
"""
import inspect
import time
import warnings

import numpy as np
from scipy import optimize

_FD_METHODS = ('2-point', '3-point')


class _MemoizeFunJac:
    """Split ``fun(x) -> (f, g)`` into separate cached ``fun`` and ``jac``."""

    def __init__(self, fun):
        self.fun = fun
        self.x = None
        self.f = None
        self.g = None

    def _compute(self, x, *args):
        if self.x is None or not np.array_equal(x, self.x):
            self.x = np.copy(x)
            f, g = self.fun(x, *args)
            self.f = f
            self.g = np.asarray(g, dtype=float)

    def value(self, x, *args):
        self._compute(x, *args)
        return self.f

    def derivative(self, x, *args):
        self._compute(x, *args)
        return self.g


def _fd_gradient(f, x, method):
    """Finite-difference gradient of ``f`` at ``x``."""
    g = np.empty_like(x)
    if method == '2-point':
        f0 = f(x)
        for i in range(x.size):
            h = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x[i]))
            xh = x.copy()
            xh[i] += h
            g[i] = (f(xh) - f0) / h
    else:
        for i in range(x.size):
            h = np.finfo(float).eps ** (1 / 3) * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def _wants_intermediate_result(callback):
    try:
        params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False
    return set(params) == {'intermediate_result'}


def minimize(fun, x0, *, args=(), jac=None, callback=None,
             maxiter=10000, gtol=1e-6, pertlim=5, disp=False,
             hess=None, hessp=None, bounds=None, constraints=None,
             **unknown_options):
    """
    Minimize a smooth function with hybrid cubic regularized conjugate gradient.

    Can be called directly, or passed to SciPy as a custom method:
    ``scipy.optimize.minimize(fun, x0, jac=grad, method=minimize)``.
    Defaults for ``maxiter``, ``gtol`` and ``pertlim`` follow Section 4 of
    Buhler, Benson & Shanno (2024).

    Parameters
    ----------
    fun : callable
        Objective function, ``fun(x, *args) -> float``.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Extra arguments passed to ``fun`` and ``jac``.
    jac : {callable, True, None, '2-point', '3-point'}, optional
        Gradient of ``fun``. A callable ``jac(x, *args) -> ndarray`` of
        shape (n,). If True, ``fun`` returns ``(f, g)``. If None or a
        finite-difference scheme, the gradient is approximated numerically
        (None uses '2-point'). Note that ``scipy.optimize.minimize`` passes
        finite-difference strings to custom methods as None, so '3-point'
        only takes effect when this function is called directly.
    callback : callable, optional
        Called after each iteration as ``callback(intermediate_result)``
        with an ``OptimizeResult`` holding ``x`` and ``fun`` (if its only
        parameter is named ``intermediate_result``), otherwise as
        ``callback(xk)``. Raise ``StopIteration`` to stop early.
    maxiter : int, optional
        Maximum number of iterations.
    gtol : float, optional
        Converged when ``||jac(x)|| <= gtol``.
    pertlim : int, optional
        Maximum number of lambda values tried for cubic regularization
        (U in Algorithm 2). When reached, a restart is done instead.
    disp : bool, optional
        Print a summary when finished.
    hess, hessp, bounds, constraints
        Accepted for compatibility with ``scipy.optimize.minimize``.
        Not used; a warning is issued if any is given.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        Fields: ``x``, ``fun``, ``jac``, ``nit``, ``nfev``, ``njev``,
        ``status``, ``success``, ``message``, ``invoked_cubic``, ``time``,
        and ``history``.

        Status codes: 0 = solved, 1 = search direction is not a descent
        direction, 2 = iterations limit reached, 3 = search direction is
        undefined, 99 = ``callback`` raised ``StopIteration``.
    """
    if unknown_options:
        warnings.warn('Unknown solver options: ' + ', '.join(unknown_options),
                      optimize.OptimizeWarning, stacklevel=2)
    if hess is not None or hessp is not None:
        warnings.warn('hybrid_cubic_cg does not use Hessian information '
                      '(hess/hessp).', RuntimeWarning, stacklevel=2)
    if bounds is not None or constraints:
        warnings.warn('hybrid_cubic_cg cannot handle bounds or constraints; '
                      'they are ignored.', RuntimeWarning, stacklevel=2)

    if not isinstance(args, tuple):
        args = (args,)

    if jac is True:
        memo = _MemoizeFunJac(fun)
        raw_fun, raw_jac = memo.value, memo.derivative
    elif callable(jac):
        raw_fun, raw_jac = fun, jac
    elif jac is None or jac is False or jac in _FD_METHODS:
        scheme = '2-point' if jac in (None, False) else jac
        raw_fun = fun
        raw_jac = lambda x, *a: _fd_gradient(fun_counted, x, scheme)
    else:
        raise ValueError(f"jac must be callable, True, None, '2-point' or "
                         f"'3-point'; got {jac!r}.")

    #initialize
    t_start = time.time()
    nfev = 0
    njev = 0

    def fun_counted(z):
        nonlocal nfev
        nfev += 1
        return float(raw_fun(z, *args))

    def jac_counted(z):
        nonlocal njev
        njev += 1
        return np.asarray(raw_jac(z, *args), dtype=float)

    new_style_callback = callback is not None and _wants_intermediate_result(callback)

    def notify(xk, fk):
        """Run the user callback on an accepted iterate. True means stop."""
        if callback is None:
            return False
        try:
            if new_style_callback:
                callback(intermediate_result=optimize.OptimizeResult(
                    x=np.copy(xk), fun=fk))
            else:
                callback(np.copy(xk))
        except StopIteration:
            return True
        return False

    #data preprocessing
    # CG solver
    x = np.asarray(x0, dtype=float).flatten()
    n = x.size
    alpha = 1.0
    c = jac_counted(x)
    pertcnt = 0
    nrst = n
    restart = False

    inPowell = False
    dprat = 0.0
    ddprat = 0.0
    lam = 0
    k = 0

    #set to None because we will define later
    c0 = None
    dprat0 = None
    prat = None
    prat0 = None
    olamb = None
    oolamb = None
    alpha0 = None
    restart0 = None
    xprev = None
    dx0 = None
    c00 = None

    status = None
    objs = []
    gradNorms = []
    history = {}
    history['objective'] = []
    history['gradNorm'] = []

    while (k < maxiter):
        k += 1
        xTx = np.dot(x,x)
        cTc = np.dot(c,c)
        retry = False

        # Check for convergence
        if (np.sqrt(cTc) <= gtol):
            status = 0 #solved
            if objs and notify(x, objs[-1]):
                status = 99
            break

        # Compute step direction
        if restart == False:
            dx = -c
        else:
            # test to see if powell restart criterion holds
            if (nrst != n) and (restart >1) and (abs(np.dot(c,c0)/cTc) > 0.2):
                if nrst == 0:
                    nrst = n
                inPowell = True
                retry = True
                if lam == 0.0:
                    if dprat0 != 0.0:
                        ddprat = ( abs(np.dot(c,c0)/cTc) - 2*prat + prat0 ) / ( (lam - olamb)*(oolamb-olamb) )
                    dprat0 = dprat
                    if lam >0:
                        dprat = ( abs(np.dot(c,c0)/cTc) - prat ) / ( lam - olamb )
                    if prat is not None:
                        prat0 = prat

                    prat = abs(np.dot(c,c0)/cTc)
                    alpha = alpha0
                    restart = restart0
                    x = xprev
                    dx = dx0
                    c = c0
                    c0 = c00
                    xTx = np.dot(x,x)
                    cTc = np.dot(c,c)

                    if olamb is not None:
                        oolamb = olamb
                    olamb = lam
                    if lam == 0.0:
                        lam = prat/0.2
                    else:
                        if pertcnt <= 0.2:
                            lam = olamb - prat/dprat
                        else:
                            if ( dprat*dprat - 2*ddprat*prat > 0) and (abs(ddprat) > 1e-8 ):
                                lam = olamb + max( (-dprat + np.sqrt(dprat*dprat - 2*ddprat*prat))/ddprat, (-dprat - np.sqrt(dprat*dprat - 2*ddprat*prat))/ddprat )
                            else:
                                lam = 2*olamb

                        if lam < 1e-12:
                            lam = 2*olamb

                else:
                    lam = 2*lam
                pertcnt += 1
                k -= 1

            else:
                dx0 = dx
                if ( pertcnt > 0 ):
                    lam = lam/2.0
                else:
                    lam = 0.0
                dprat = 0.0
                dprat0 = 0.0
                prat0 = 0.0
                pertcnt = 0
            # If performing a restart, update the Beale restart vectors
            if ( nrst == n ):
                pt = alpha*dx
                yt = c - c0
                ytTyt = np.dot(yt,yt)
                cTyt = np.dot(pt,yt)
                cTct = np.dot(pt,pt)

            p = alpha*dx
            y = c - c0
            pTc = np.dot(pt,c)
            yTc = np.dot(yt,c)

            u1 = -pTc/ytTyt
            u2 = 2*pTc/cTyt - yTc/ytTyt
            u3 = cTyt/ytTyt

            if ( pertcnt == 0 ) :
                dx = -u3*c - u1*yt - u2*pt
            else:
                bracket = lam*lam + 2*lam*ytTyt/cTyt + ytTyt/cTct
                a = -cTyt/(lam*cTyt + ytTyt)
                b1 = (-lam*cTyt - 2*ytTyt)*ytTyt/(cTyt*cTct*bracket*(lam*cTyt + ytTyt))
                d = lam/(bracket*(lam*cTyt + ytTyt))
                e = ytTyt/(cTct*bracket*(lam*cTyt + ytTyt))

                bracket = lam*lam+lam/u3+lam*ytTyt/cTyt+cTyt/(u3*cTct)
                denom = u3*u3*cTct*lam*bracket + u3*cTct*bracket
                d = u3*u3*lam*(cTct/cTyt) / denom

                dx = a*c + b1*pTc*pt + d*yTc*yt + e*yTc*pt + e*pTc*yt

            if ( nrst != n ):
                if ( pertcnt == 0 ):
                    u1 = -np.dot(y,pt)/ytTyt
                    u2 = -np.dot(y,yt)/ytTyt + 2*np.dot(y,pt)/cTyt
                    u3 = np.dot(p,y)
                    temp = cTyt/ytTyt*y + u1*yt + u2*pt
                    u4 = np.dot(temp, y)

                    u1 = -np.dot(p,c)/u3
                    u2 = (u4/u3 + 1)*np.dot(p,c)/u3 - np.dot(c,temp)/u3
                    dx = dx - u1*temp - u2*p
                else:
                    ptTy = np.dot(pt,y)
                    ytTy = np.dot(yt,y)
                    temp1 = -(a*y + b1*ptTy*pt + d*ytTy*yt + e*ytTy*pt + e*ptTy*yt)

                    a2 = 1.0/(lam*u3+1.0)
                    b2 = lam*b1
                    d2 = lam*d
                    e2 = lam*e
                    ptTp = np.dot(pt,p)
                    ytTp = np.dot(yt,p)
                    temp2 = a2*p + b2*ptTp*pt + d2*ytTp*yt + e2*ytTp*pt + e2*ptTp*yt

                    u10 = np.dot(temp1, c)
                    u11 = np.dot(temp2, c)
                    u12 = np.dot(temp1, y)
                    u13 = np.dot(temp2, y)
                    u14 = np.dot(temp2, p)
                    u15 = np.dot(p, y)
                    denom = lam*u14*(u15 + u12) + u13*u13
                    if denom < np.finfo(float).eps:
                        status = 3
                        break
                    dx = dx + lam*u14*u10*temp1/denom - (u15 + u12)*u11*temp2/denom + u13*u10*temp2/denom + u13*u11*temp1/denom

        # The previous step is accepted unless the Powell test sent us back
        if not retry and objs and notify(x, objs[-1]):
            status = 99
            break

        # Check that the search direction is a descent direction
        dxTc = np.dot(dx, c)
        if ( dxTc > 0 ):
            status = 1
            break
        # Save the current point
        xprev = x
        if c0 is not None:
            c00 = c0
        c0 = c
        alpha0 = alpha
        restart0 = restart

        if ( restart == 0 ):
            restart = 1
        else:
            if ( pertcnt == 0 ):
                if ( nrst == n ) :
                    nrst = 0
                nrst +=  1
                restart = 2

        def afind(a):
            return fun_counted(x + a*dx)
        alpha = optimize.fminbound(afind, 0, 10)

        # Take the step and update function value and gradient
        x = xprev + alpha*dx
        c = jac_counted(x)
        objs.append(fun_counted(x))
        gradNorms.append(np.linalg.norm(c))

    elapsedTime = time.time() - t_start

    if status is None:
        status = 2

    messages = {
        0: 'Optimization terminated successfully.',
        1: 'Search direction is not a descent direction.',
        2: 'Iterations limit reached.',
        3: 'Search direction is undefined.',
        99: '`callback` raised `StopIteration`.',
    }

    if disp:
        print(messages[status])
        print('time elapsed: ' + str(elapsedTime))
        print('n = %d, Iters = %d, invokedCubic = %s\n'% (n, k, inPowell == 1))

    history['objective'] = objs
    history['gradNorm'] = gradNorms
    history['status'] = status
    history['time'] = elapsedTime
    history['iters'] = k
    history['invokedCubic'] = inPowell == 1

    return optimize.OptimizeResult(
        x=x,
        fun=fun_counted(x),
        jac=c,
        nit=k,
        nfev=nfev,
        njev=njev,
        status=status,
        success=(status == 0),
        message=messages[status],
        invoked_cubic=bool(inPowell),
        time=elapsedTime,
        history=history,
    )
