"""
Stratum-conditional maximum likelihood estimator for event studies with
selection on the outcome (Section 2.8 of Mankad & Schmidt, "Sample Selection on the Outcome with an Application to Event Studies").

Model (reduced form, Section 2 of the paper):
    AR_i = gamma + eps_i,   eps_i ~ tau * t_nu  (or N(0, tau^2))
    C_i  = 1{ b0 + b1*AR_i^+ + b2*AR_i^- + x_i'b3 + v_i > 0 },  v_i ~ N(0,1) indep.
Sampling: every covered event (C=1) plus a random sample of uncovered events (C=0).
Likelihood: sum_i log f(AR_i | x_i, C_i; theta)  -- valid under choice-based sampling.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import jax, jax.numpy as jnp
from jax.scipy.special import ndtr, log_ndtr, gammaln
from functools import partial
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import statsmodels.api as sm
import warnings
jax.config.update("jax_enable_x64", True)

_t, _w = np.polynomial.legendre.leggauss(80)
_T, _W = jnp.array((_t + 1) / 2), jnp.array(_w / 2)
KAPPA_BOUNDS = (-4.0, 7.0)          # nu = 2 + exp(kappa): nu in (2.02, 1099)

def _logdens(z, nu, dist):
    if dist == "normal":
        return -0.5 * z**2 - 0.5 * jnp.log(2 * jnp.pi)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi)
            - (nu + 1) / 2 * jnp.log1p(z**2 / nu))

def _P1(th, xb, tau, dist):
    """P(C=1|x) = int Phi(b0 + b1 a+ + b2 a- + x'b3) g(a-gamma; tau(x)) da, per observation.
    Change of variables a = gamma + tau*tan(w); Gauss-Legendre on each side of the kink a=0."""
    g, nu, b0, b1, b2 = th[0], 2 + jnp.exp(th[2]), th[3], th[4], th[5]
    w0 = jnp.arctan(-g / tau)                                   # (n,)
    out = 0.0
    for lo, hi in ((jnp.full_like(w0, -jnp.pi / 2), w0), (w0, jnp.full_like(w0, jnp.pi / 2))):
        w = lo[:, None] + (hi - lo)[:, None] * _T[None, :]
        z = jnp.tan(w)
        a = g + tau[:, None] * z
        dens = jnp.exp(_logdens(z, nu, dist)) / jnp.cos(w)**2 * (hi - lo)[:, None] * _W[None, :]
        I = b0 + b1 * jnp.maximum(a, 0) + b2 * jnp.maximum(-a, 0) + xb[:, None]
        out = out + (ndtr(I) * dens).sum(1)
    return out

@partial(jax.jit, static_argnums=5)
def _nll(th, y, X, S, C, dist):
    k, m = X.shape[1], S.shape[1]
    g, nu = th[0], 2 + jnp.exp(th[2])
    ls = th[1] + S @ th[6 + k:6 + k + m]          # log scale, optionally depending on covariates S
    tau = jnp.exp(ls)
    xb = X @ th[6:6 + k]
    idx = th[3] + th[4] * jnp.maximum(y, 0) + th[5] * jnp.maximum(-y, 0) + xb
    p = jnp.clip(_P1(th, xb, tau, dist), 1e-12, 1 - 1e-12)
    ll = (_logdens((y - g) / tau, nu, dist) - ls
          + jnp.where(C == 1, log_ndtr(idx) - jnp.log(p), log_ndtr(-idx) - jnp.log1p(-p)))
    return -ll.sum()

_grad = jax.jit(jax.grad(_nll), static_argnums=5)
_hess = jax.jit(jax.hessian(_nll), static_argnums=5)

def _optimize(th0, y, X, S, C, dist, fix_gamma=None):
    bounds = [(None, None)] * len(th0)
    bounds[2] = KAPPA_BOUNDS if dist == "t" else (0.0, 0.0)
    th0 = np.array(th0, float)
    if dist != "t":
        th0[2] = 0.0
    if fix_gamma is not None:
        bounds[0] = (fix_gamma, fix_gamma); th0[0] = fix_gamma
    f = lambda t: float(_nll(t, y, X, S, C, dist))
    gr = lambda t: np.asarray(_grad(t, y, X, S, C, dist))
    r = minimize(f, th0, jac=gr, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7})
    return r

def fit_smle(AR, X, C, dist="t", lr_nulls=(), scale_X=None):
    """Fit the stratum-conditional MLE.
    AR : (n,) abnormal returns;  X : (n,k) coverage covariates (k>=0);  C : (n,) 0/1 coverage.
    dist : 't' (default; reverts to 'normal' if nu hits its upper bound) or 'normal'.
    lr_nulls : values gamma0 at which to compute likelihood-ratio p-values.
    scale_X : optional (n,m) covariates for the outcome scale, log tau_i = log tau + scale_X_i'd
              (heteroskedastic errors); None = common scale.
    Returns dict with gamma, se, p (Wald, H0: gamma=0), nu, dist, ok, loglik, lr_p {gamma0: p}."""
    AR = np.asarray(AR, float); C = np.asarray(C, float)
    X = np.zeros((len(AR), 0)) if X is None else np.asarray(X, float).reshape(len(AR), -1)
    s0 = AR.std()
    y = AR / s0
    sd = X.std(0); sd[sd == 0] = 1.0
    Xs = (X - X.mean(0)) / sd
    S = np.zeros((len(AR), 0)) if scale_X is None else np.asarray(scale_X, float).reshape(len(AR), -1)
    sds = S.std(0); sds[sds == 0] = 1.0
    Ss = (S - S.mean(0)) / sds
    Z = np.column_stack([np.ones_like(y), np.maximum(y, 0), np.maximum(-y, 0), Xs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sm.Probit(C, Z).fit(disp=0, maxiter=200).params
    th0 = np.r_[y.mean(), np.log(y.std() * (0.8 if dist == "t" else 1.0)), np.log(3.0), b, np.zeros(Ss.shape[1])]
    yj, Xj, Sj, Cj = jnp.array(y), jnp.array(Xs), jnp.array(Ss), jnp.array(C)
    r = _optimize(th0, yj, Xj, Sj, Cj, dist)
    used = dist
    if dist == "t" and r.x[2] >= KAPPA_BOUNDS[1] - 1e-3:       # data look normal
        r = _optimize(r.x, yj, Xj, Sj, Cj, "normal"); used = "normal"
    th = r.x
    free = [i for i in range(len(th)) if not (i == 2 and used == "normal")]
    try:
        H = np.asarray(_hess(jnp.array(th), yj, Xj, Sj, Cj, used))[np.ix_(free, free)]
        V = np.linalg.inv(H)
        se = float(np.sqrt(V[0, 0])) * s0 if V[0, 0] > 0 else np.nan
    except Exception:
        se = np.nan
    gam = float(th[0]) * s0
    out = dict(gamma=gam, se=se, p=2 * norm.sf(abs(gam / se)) if se == se else np.nan,
               nu=float(2 + np.exp(th[2])) if used == "t" else np.inf, dist=used,
               ok=bool(r.success), loglik=-float(r.fun), b=th[3:6].tolist(), d=th[6 + X.shape[1]:].tolist(), lr_p={})
    for g0 in lr_nulls:
        rc = _optimize(th, yj, Xj, Sj, Cj, used, fix_gamma=g0 / s0)
        lr = max(0.0, 2 * (rc.fun - r.fun))
        out["lr_p"][g0] = float(chi2.sf(lr, 1))
    return out
