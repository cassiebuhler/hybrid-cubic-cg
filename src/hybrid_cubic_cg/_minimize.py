"""
Nonlinear conjugate gradient with hybrid cubic regularization.

Implements the Hybrid Cubic Regularization of CGM from
Buhler, Benson & Shanno (2024), "Regularized step directions in nonlinear
conjugate gradient methods", Mathematical Programming Computation 16(4).
"""
import time

import numpy as np
from scipy import optimize

def minimize(fun, x0, jac, *, maxiter=10000, gtol=1e-6, pertlim=5, disp=False):
    """
    Minimize a smooth function with hybrid cubic regularized conjugate gradient.

    Parameters
    ----------
    fun : callable
        Objective function, ``fun(x) -> float``.
    x0 : array_like, shape (n,)
        Initial guess.
    jac : callable
        Gradient of ``fun``, ``jac(x) -> ndarray`` of shape (n,).
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
        undefined.
    """
    #initialize
    t_start = time.time()
    nfev = 0
    njev = 0

    #data preprocessing
    # CG solver
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    alpha = 1.0
    c = jac(x)
    njev += 1
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

        # Check for convergence
        if (np.sqrt(cTc) <= gtol):
            status = 0 #solved
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
            nonlocal nfev
            nfev += 1
            return fun(x + a*dx)
        alpha = optimize.fminbound(afind, 0, 10)

        # Take the step and update function value and gradient
        x = xprev + alpha*dx
        c = jac(x)
        njev += 1
        objs.append(fun(x))
        nfev += 1
        gradNorms.append(np.linalg.norm(c))

    elapsedTime = time.time() - t_start

    if status is None:
        status = 2

    messages = {
        0: 'Optimization terminated successfully.',
        1: 'Search direction is not a descent direction.',
        2: 'Iterations limit reached.',
        3: 'Search direction is undefined.',
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
        fun=fun(x),
        jac=c,
        nit=k,
        nfev=nfev + 1,
        njev=njev,
        status=status,
        success=(status == 0),
        message=messages[status],
        invoked_cubic=bool(inPowell),
        time=elapsedTime,
        history=history,
    )
