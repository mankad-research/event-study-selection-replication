"""
Estimators used in the paper (all except the likelihood estimator, which is in smle.py).

  none_mean            uncorrected mean of the combined sample
  q_weighted_mean      proposed estimator when Q is known or estimated (Section 2.6, eq. qmean/qse; Proposition 3)
  heckman_twostep      textbook Heckman (1979) two-step with corrected standard errors (Section 3 comparator)
"""
import numpy as np, statsmodels.api as sm, warnings
from scipy.stats import norm

def none_mean(AR):
    AR = np.asarray(AR, float); se = AR.std(ddof=1) / np.sqrt(len(AR)); g = AR.mean()
    return dict(gamma=g, se=se, p=2 * norm.sf(abs(g / se)))

def q_weighted_mean(AR, C, Q, m=None):
    """gamma_Q = Q*mean(covered) + (1-Q)*mean(supplement).  If Q was estimated as the covered share of a random
    draw of m events from the non-discretionary channel, pass m to add (mean1-mean0)^2 Q(1-Q)/m to the variance."""
    AR, C = np.asarray(AR, float), np.asarray(C, float)
    a1, a0 = AR[C == 1], AR[C == 0]
    g = Q * a1.mean() + (1 - Q) * a0.mean()
    v = Q**2 * a1.var(ddof=1) / len(a1) + (1 - Q)**2 * a0.var(ddof=1) / len(a0)
    if m is not None:
        v += (a1.mean() - a0.mean())**2 * Q * (1 - Q) / m
    se = np.sqrt(v)
    return dict(gamma=g, se=se, p=2 * norm.sf(abs(g / se)))

# ---------------------------------------------------------------- textbook Heckman comparator
def heckman_twostep(AR, X, C):
    """Probit of C on (1, x) in the combined sample; OLS of AR on (1, inverse Mills ratio) in the covered
    sample; Heckman's (1979) corrected two-step standard errors.  Returns dict(gamma, se, p)."""
    AR = np.asarray(AR, float); C = np.asarray(C, float)
    W = sm.add_constant(np.asarray(X, float).reshape(len(AR), -1), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = sm.Probit(C, W).fit(disp=0, maxiter=200)
    c, Vc = pr.params, pr.cov_params()
    s = C == 1; Ws = W[s]; zc = Ws @ c
    lam = norm.pdf(zc) / np.clip(norm.cdf(zc), 1e-300, None); delta = lam * (lam + zc)
    Xst = np.column_stack([np.ones(s.sum()), lam]); XtXi = np.linalg.inv(Xst.T @ Xst)
    bb = XtXi @ Xst.T @ AR[s]; e = AR[s] - Xst @ bb
    sig2 = e @ e / s.sum() + bb[1]**2 * delta.mean(); rho2 = min(bb[1]**2 / sig2, 1.0)
    F = (Xst * delta[:, None]).T @ Ws
    V = sig2 * XtXi @ (Xst.T @ ((1 - rho2 * delta)[:, None] * Xst) + rho2 * F @ Vc @ F.T) @ XtXi
    se = float(np.sqrt(V[0, 0]))
    return dict(gamma=float(bb[0]), se=se, p=float(2 * norm.sf(abs(bb[0] / se))))
