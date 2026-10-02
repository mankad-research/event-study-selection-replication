"""Check Propositions 1-2 (as typeset) and Lemma 1 against Monte Carlo. Runtime ~2 minutes."""
import numpy as np
from scipy.stats import norm, multivariate_normal as mvn

def Phi2(h, k, r): return mvn(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([h, k])
def Phibar2(h, k, r): return 1 - norm.cdf(h) - norm.cdf(k) + Phi2(h, k, r)
def Psi(h, k, r):
    s = np.sqrt(1 - r * r)
    return norm.pdf(h) * (1 - norm.cdf((k - r * h) / s)) + r * norm.pdf(k) * (1 - norm.cdf((h - r * k) / s))
def LU(b0, b1, b2, b3, g, x): return (b0 + b1 * g + b3 * x) / (-b1), (b0 - b2 * g + b3 * x) / b2

def prop1(b0, b1, b2, b3, g, se, rho, x):
    sU = np.sqrt(se**2 + b2**-2 - 2 * rho * se / b2); sL = np.sqrt(se**2 + b1**-2 + 2 * rho * se / b1)
    sUL = -se**2 + rho * se * (1 / b2 - 1 / b1) + 1 / (b1 * b2); rUL = sUL / (sU * sL)
    L0, U0 = LU(b0, b1, b2, b3, g, x); hU, hL = U0 / sU, -L0 / sL
    c = np.linalg.solve([[sU**2, sUL], [sUL, sL**2]], [se**2 - rho * se / b2, -se**2 - rho * se / b1])
    return -(c[0] * sU * Psi(hU, hL, rUL) + c[1] * sL * Psi(hL, hU, rUL)) / (1 - Phibar2(hU, hL, rUL))

def prop2(b0, b1, b2, b3, g, se, rho, x):
    sU = np.sqrt(se**2 + b2**-2 - 2 * rho * se / b2); sL = np.sqrt(se**2 + b1**-2 + 2 * rho * se / b1)
    u0 = -(b0 + b3 * x); L0, U0 = LU(b0, b1, b2, b3, g, x)
    ruL, ruU = (-rho * se - 1 / b1) / sL, (rho * se - 1 / b2) / sU
    hA, kA, hB, kB = -u0, -L0 / sL, u0, -U0 / sU
    d = np.linalg.solve([[1, -ruL], [-ruL, 1]], [-rho * se, (-se**2 - rho * se / b1) / sL])
    e = np.linalg.solve([[1, -ruU], [-ruU, 1]], [rho * se, -(se**2 - rho * se / b2) / sU])
    num = d[0] * Psi(hA, kA, -ruL) + d[1] * Psi(kA, hA, -ruL) + e[0] * Psi(hB, kB, -ruU) + e[1] * Psi(kB, hB, -ruU)
    return -num / (1 - Phibar2(hA, kA, -ruL) - Phibar2(hB, kB, -ruU))

def bias(b0, b1, b2, b3, g, se, rho, x):
    return prop1(b0, b1, b2, b3, g, se, rho, x) if b2 > 0 else prop2(b0, b1, b2, b3, g, se, rho, x)

def mc(b0, b1, b2, b3, g, se, rho, x, N=4_000_000, seed=0):
    r = np.random.default_rng(seed); e = r.normal(size=N); v = r.normal(size=N)
    u = rho * e + np.sqrt(1 - rho**2) * v; eps = se * e; A = g + eps
    C = b0 + b1 * np.maximum(A, 0) + b2 * np.maximum(-A, 0) + b3 * x + u > 0
    return eps[C].mean(), eps[C].std() / np.sqrt(C.sum())

if __name__ == "__main__":
    print("Propositions 1-2: formula vs Monte Carlo (z should be within +-2)")
    for c in [(-5, 300, 100, 4, -0.02, 0.05, 0.0, 1.0), (-5, 100, 300, 4, 0.02, 0.05, 0.3, 0.5),
              (-2, 40, 60, 1, 0.01, 0.05, -0.6, 0.0), (-1, 20, 10, 1, 0.0, 0.1, 0.7, -0.5),
              (-5, 300, -50, 4, -0.02, 0.05, 0.0, 1.0), (-2, 40, -20, 1, 0.01, 0.05, 0.5, 0.0),
              (0.3, 30, -60, 0.5, 0.02, 0.05, 0.6, 0.3)]:
        f = bias(*c); m, s = mc(*c)
        print(f"  {'P1' if c[2] > 0 else 'P2'} {c}: formula {f:+.6f}  MC {m:+.6f}  z={(f - m) / s:+.2f}")
    print("Lemma 1: bias at (beta, rho) equals bias at reduced-form (b, 0)")
    for c in [(-5, 300, 100, 4, -0.02, 0.05, 0.4, 1.0), (-1, 20, 10, 1, 0.0, 0.1, 0.7, -0.5), (-2, 40, 10, 1, 0.01, 0.05, 0.8, 0.3)]:
        b0, b1, b2, b3, g, se, rho, x = c; s = np.sqrt(1 - rho**2)
        b = ((b0 - rho * g / se) / s, (b1 + rho / se) / s, (b2 - rho / se) / s, b3 / s)
        print(f"  rho={rho}: structural {bias(*c):+.6f}  reduced form {bias(*b, g, se, 0.0, x):+.6f}")
    print("Limit beta2 -> 0: Prop 1 (b2>0) and Prop 2 (b2<0) meet; classical IMR only when b1=b2=0")
    for b2 in [0.1, 0.01, 0.001]:
        print(f"  |b2|={b2}: P1 {prop1(-1, 20, b2, 1, 0.0, 0.1, 0.5, -0.5):+.6f}  P2 {prop2(-1, 20, -b2, 1, 0.0, 0.1, 0.5, -0.5):+.6f}")
