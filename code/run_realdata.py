"""
Application of Section 4 (Figure 2, Table 3, footnote with winsorized returns, and the numbers quoted in Section 4).

    python run_realdata.py --workers 8
Reads ../data/working_data_dedup.pkl.  The outcome is abret_10, the abnormal return around the press-release date,
for every event (covered or not).  Holding the WSJ-covered events fixed, draws random subsamples of the
press-release-only pool at each supplement ratio and applies every estimator; also computes full-sample benchmarks.
Writes ../output/realdata_reps_{raw,winsor}.csv and ../output/realdata_full_{raw,winsor}.json.  Resumes if interrupted.
"""
import argparse, os, sys, json, csv, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

COVAR_COLS = ["Female", "logMV", "Outsider", "Analyst"]
RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
FIELDS = ["ratio", "rep", "n_sup", "none_est", "heck_est", "heck_se", "qw_est", "qw_se",
          "mle_est", "mle_se", "mle_nu", "mle_dist", "mle_ok", "mle_lr_p_zero"]

def load(pkl, outcome, winsor):
    from scipy.stats import mstats
    df = pd.read_pickle(pkl)
    AR = df[outcome].values.astype(float)
    if winsor > 0:
        AR = np.asarray(mstats.winsorize(AR, limits=(winsor, winsor)), float)
    return AR, df[COVAR_COLS].values.astype(float), df.C.values.astype(float)

def one_task(pkl, outcome, winsor, ratio, rep, seed, mle=True):
    from estimators import none_mean, q_weighted_mean, heckman_twostep
    AR0, X0, C0 = load(pkl, outcome, winsor); Q = C0.mean()
    cov, pool = np.where(C0 == 1)[0], np.where(C0 == 0)[0]
    rng = np.random.default_rng([seed, int(round(ratio * 1000)), rep, int(winsor * 1000)])
    n_sup = int(round(ratio * len(cov)))
    idx = np.r_[cov, rng.choice(pool, n_sup, replace=False)] if n_sup > 0 else cov
    AR, X, C = AR0[idx], X0[idx], C0[idx]
    out = dict(ratio=ratio, rep=rep, n_sup=n_sup, none_est=AR.mean())
    if n_sup < 10:
        return out
    r = q_weighted_mean(AR, C, Q); out.update(qw_est=r["gamma"], qw_se=r["se"])
    try:
        r = heckman_twostep(AR, X, C); out.update(heck_est=r["gamma"], heck_se=r["se"])
    except Exception:
        pass
    if mle:
        from smle import fit_smle
        try:
            r = fit_smle(AR, X, C, dist="t", lr_nulls=(0.0,))
            out.update(mle_est=r["gamma"], mle_se=r["se"], mle_nu=r["nu"], mle_dist=r["dist"], mle_ok=int(r["ok"]), mle_lr_p_zero=r["lr_p"][0.0])
        except Exception:
            out["mle_ok"] = 0
    return out

def full_sample(pkl, outcome, winsor, mle=True):
    import statsmodels.api as sm
    AR, X, C = load(pkl, outcome, winsor); Q = C.mean()
    Z = sm.add_constant(np.column_stack([np.maximum(AR, 0), np.maximum(-AR, 0), X]))
    pr = sm.Probit(C, Z).fit(disp=0, maxiter=200)
    names = ["const", "AR_pos", "AR_neg"] + COVAR_COLS
    s1, s0, n1 = AR[C == 1].std(ddof=1), AR[C == 0].std(ddof=1), int(C.sum())
    out = dict(outcome=outcome, winsor=winsor, n=len(AR), n_covered=n1, Q=Q,
               full_mean=float(AR.mean()), full_se=float(AR.std(ddof=1) / np.sqrt(len(AR))), full_median=float(np.median(AR)),
               covered_mean=float(AR[C == 1].mean()), uncovered_mean=float(AR[C == 0].mean()),
               qw_se_by_ratio={str(r): float(np.sqrt(Q**2 * s1**2 / n1 + (1 - Q)**2 * s0**2 / round(r * n1))) for r in RATIOS if r > 0},
               probit={n: dict(coef=float(b), se=float(s), p=float(p)) for n, b, s, p in zip(names, pr.params, pr.bse, pr.pvalues)})
    if mle:
        from smle import fit_smle
        r = fit_smle(AR, X, C, dist="t"); out.update(mle_full=r["gamma"], mle_full_se=r["se"], mle_full_nu=r["nu"])
    return out

def run(pkl, outcome, winsor, tag, reps, workers, seed, mle):
    out = os.path.join("..", "output", f"realdata_reps_{tag}.csv"); os.makedirs(os.path.dirname(out), exist_ok=True)
    cfg = dict(outcome=outcome, winsor=winsor, seed=seed, mle=mle, reps=reps); cfg_file = out + ".config.json"; done = set()
    if os.path.exists(out):
        old = json.load(open(cfg_file)) if os.path.exists(cfg_file) else None
        if old != cfg:
            sys.exit(f"{out} was produced with different settings ({old}); delete it to start over.")
        done = {(float(r["ratio"]), int(r["rep"])) for r in csv.DictReader(open(out))}
    else:
        json.dump(cfg, open(cfg_file, "w"))
    tasks = [(r, k) for r in RATIOS for k in range(reps if r > 0 else 1) if (r, k) not in done]
    print(f"[{tag}] {len(tasks)} tasks to run", flush=True)
    new = not os.path.exists(out)
    f = open(out, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        ff = ex.submit(full_sample, pkl, outcome, winsor, mle)
        futs = [ex.submit(one_task, pkl, outcome, winsor, r, k, seed, mle) for r, k in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            w.writerow(fu.result()); f.flush()
            if i % 100 == 0 or i == len(futs):
                print(f"  [{tag}] {i}/{len(futs)} ({(time.time()-t0)/60:.1f} min)", flush=True)
        json.dump(ff.result(), open(os.path.join("..", "output", f"realdata_full_{tag}.json"), "w"), indent=1)
    f.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data", default=os.path.join("..", "data", "working_data_dedup.pkl"))
    ap.add_argument("--outcome", default="abret_10")
    ap.add_argument("--winsor-level", type=float, default=0.02)
    ap.add_argument("--no-mle", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.data):
        sys.exit(f"Data file not found: {a.data}. See ../data/README.md.")
    run(a.data, a.outcome, 0.0, "raw", a.reps, a.workers, a.seed, not a.no_mle)
    run(a.data, a.outcome, a.winsor_level, "winsor", a.reps, a.workers, a.seed, not a.no_mle)
