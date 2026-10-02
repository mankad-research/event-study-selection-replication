"""
Monte Carlo study of Section 3 (Figure 1, Tables 1-2, and the simulation numbers quoted in Sections 2.6 and 3).

    python run_simulations.py --workers 8            # full run (paper defaults: 400 reps, 800 at the 20% supplement)
    python run_simulations.py --workers 4 --quick    # smoke test, a few minutes
Each replication is written to ../output/sim_reps.csv as soon as it finishes; rerunning the same command resumes.
"""
import argparse, os, sys, time, csv, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

LABELS = {"Prefers positive": (300, 100), "Prefers negative": (100, 300), "Indifferent": (200, 200)}
GAMMAS = [-0.02, 0.0, 0.02]
RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
TABLE_RATIO, T_DFS = 0.20, [10, 5, 3]
B0, B3, SIGMA, N_ORIG, POP_MULT = -5.0, 4.0, 0.05, 600, 8
FIELDS = ["task_id", "design", "label", "gamma", "ratio", "eps", "rep", "Q", "n_sup",
          "none_est", "none_se", "none_p", "heck_est", "heck_se", "heck_p", "qw_est", "qw_se", "qw_p",
          "mle_est", "mle_se", "mle_p", "mle_nu", "mle_dist", "mle_ok",
          "mle_lr_p_true", "mle_lr_p_zero", "comp_mean"]

def build_tasks(reps, reps_table):
    tasks, tid = [], 0
    for lab in LABELS:
        for g in GAMMAS:
            for r in RATIOS:
                for k in range(reps_table if abs(r - TABLE_RATIO) < 1e-9 else reps):
                    tasks.append((tid, "main", lab, g, r, "normal", k)); tid += 1
    for df in T_DFS:
        for lab in LABELS:
            for k in range(reps_table):
                tasks.append((tid, "tdist", lab, 0.0, TABLE_RATIO, f"t{df}", k)); tid += 1
    return tasks

def one_task(task, seed, mle=True, lr=True):
    from estimators import none_mean, q_weighted_mean, heckman_twostep
    tid, design, lab, gamma, ratio, eps, k = task
    rng = np.random.default_rng([seed, tid])
    b1, b2 = LABELS[lab]; N = N_ORIG * POP_MULT
    x = rng.normal(size=N); u = rng.normal(size=N)
    e = rng.normal(size=N) if eps == "normal" else rng.standard_t(int(eps[1:]), N) / np.sqrt(int(eps[1:]) / (int(eps[1:]) - 2))
    A = gamma + SIGMA * e
    cov = B0 + b1 * np.maximum(A, 0) + b2 * np.maximum(-A, 0) + B3 * x + u > 0
    Q = cov.mean()                                   # population coverage rate (known in this design)
    i1 = np.where(cov)[0][:N_ORIG]; n_sup = int(round(ratio * N_ORIG)); i0 = np.where(~cov)[0][:n_sup]
    idx = np.r_[i1, i0]; AR, X, C = A[idx], x[idx], np.r_[np.ones(len(i1)), np.zeros(len(i0))]
    out = dict(task_id=tid, design=design, label=lab, gamma=gamma, ratio=ratio, eps=eps, rep=k, Q=Q, n_sup=n_sup)
    r = none_mean(AR); out.update(none_est=r["gamma"], none_se=r["se"], none_p=r["p"])
    if n_sup < 10:
        return out
    out["comp_mean"] = AR[C == 0].mean()
    r = q_weighted_mean(AR, C, Q); out.update(qw_est=r["gamma"], qw_se=r["se"], qw_p=r["p"])
    try:
        r = heckman_twostep(AR, X, C); out.update(heck_est=r["gamma"], heck_se=r["se"], heck_p=r["p"])
    except Exception:
        pass
    if mle:
        from smle import fit_smle
        try:
            nulls = (((gamma,) if gamma == 0 else (gamma, 0.0)) if lr else ())
            r = fit_smle(AR, X, C, dist="t", lr_nulls=nulls)
            out.update(mle_est=r["gamma"], mle_se=r["se"], mle_p=r["p"], mle_nu=r["nu"], mle_dist=r["dist"],
                       mle_ok=int(r["ok"]), mle_lr_p_true=r["lr_p"].get(gamma, np.nan), mle_lr_p_zero=r["lr_p"].get(0.0, np.nan))
        except Exception:
            out["mle_ok"] = 0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--reps", type=int, default=400, help="replications per design and supplement size")
    ap.add_argument("--reps-table", type=int, default=800, help="replications at the 20%% supplement (Table 1, Panel B)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 / 4 replications")
    ap.add_argument("--no-mle", action="store_true", help="skip the likelihood estimator (much faster)")
    ap.add_argument("--no-lr", action="store_true", help="skip likelihood-ratio tests for the likelihood estimator")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick: a.reps, a.reps_table = 2, 4
    out = a.out or os.path.join("..", "output", "sim_reps_quick.csv" if a.quick else "sim_reps.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cfg = dict(reps=a.reps, reps_table=a.reps_table, seed=a.seed, mle=not a.no_mle, lr=not a.no_lr)
    cfg_file = out + ".config.json"; done = set()
    if os.path.exists(out):
        old = json.load(open(cfg_file)) if os.path.exists(cfg_file) else None
        if old != cfg:
            sys.exit(f"{out} was produced with different settings ({old}); delete it to start over.")
        done = {int(r["task_id"]) for r in csv.DictReader(open(out))}
    else:
        json.dump(cfg, open(cfg_file, "w"))
    tasks = [t for t in build_tasks(a.reps, a.reps_table) if t[0] not in done]
    print(f"{len(tasks)} tasks to run on {a.workers} workers ({len(done)} already done)", flush=True)
    new = not os.path.exists(out)
    f = open(out, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(one_task, t, a.seed, not a.no_mle, not a.no_lr) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                w.writerow(fu.result()); f.flush()
            except Exception as e:
                print("task failed:", e, file=sys.stderr)
            if i % 500 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i}/{len(futs)} done, {el/60:.1f} min elapsed, ~{el/i*(len(futs)-i)/60:.0f} min left", flush=True)
    f.close()

if __name__ == "__main__":
    main()
