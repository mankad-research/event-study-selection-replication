"""
Build every table and figure in the paper, plus numbers_for_text.md (every simulation and real-data number quoted
in the text, keyed to the section that uses it), from the files written by run_simulations.py and run_realdata.py.

    python make_outputs.py              # uses ../output/sim_reps.csv
    python make_outputs.py --quick      # uses ../output/sim_reps_quick.csv
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm

OUT = os.path.join("..", "output")
LAB = ["Prefers positive", "Prefers negative", "Indifferent"]
LONG = {"Prefers positive": "Favors positive deviations", "Prefers negative": "Favors negative deviations", "Indifferent": "Indifferent to direction"}
FIG = {"Prefers positive": "Prefers positive returns", "Prefers negative": "Prefers negative returns", "Indifferent": "Indifferent to direction"}
COL = {"none": "#4C72B0", "heck": "#DD8452", "qw": "#55A868", "mle": "#C44E52"}
MK = {"none": "o", "heck": "^", "qw": "s", "mle": "v"}
NM = {"none": "None", "heck": "Heckman", "qw": "Proposed", "mle": "Likelihood"}
Z = 1.959964
p = lambda *a: os.path.join(OUT, *a)
rej = lambda s: (pd.Series(s).dropna() < 0.05).mean()
pct = lambda v: f"{100*v:.1f}%"

def summarize(m):
    rows = []
    for (lab, g, r), x in m.groupby(["label", "gamma", "ratio"]):
        row = dict(label=lab, gamma=g, ratio=r, n=len(x))
        for e in ["none", "heck", "qw", "mle"]:
            v = x[f"{e}_est"].dropna() if f"{e}_est" in x else pd.Series(dtype=float)
            row[f"{e}_mean"], row[f"{e}_sd"] = (v.mean(), v.std(ddof=1)) if len(v) else (np.nan, np.nan)
        for e in ["qw", "mle"]:
            ok = x.dropna(subset=[f"{e}_est", f"{e}_se"])
            row[f"{e}_cov"] = (np.abs(ok[f"{e}_est"] - g) / ok[f"{e}_se"] < Z).mean() if len(ok) else np.nan
            row[f"{e}_se_mean"] = ok[f"{e}_se"].mean() if len(ok) else np.nan
        row["mle_cov_lr"] = (x.mle_lr_p_true.dropna() > 0.05).mean() if x.mle_lr_p_true.notna().any() else np.nan
        row["comp_mean"] = x.comp_mean.mean()
        rows.append(row)
    return pd.DataFrame(rows)

def figure1(S):
    fig, ax = plt.subplots(3, 3, figsize=(7.75, 6), sharex=True)
    for i, lab in enumerate(LAB):
        for j, g in enumerate(GAM):
            a = ax[i, j]; s0 = S[(S.label == lab) & np.isclose(S.gamma, g)].sort_values("ratio")
            for e in ["none", "heck", "qw"]:
                s = s0.dropna(subset=[f"{e}_mean"])
                a.plot(s.ratio, s[f"{e}_mean"], marker=MK[e], color=COL[e], label=NM[e], ms=4, lw=1.3)
                a.fill_between(s.ratio, s[f"{e}_mean"] - s[f"{e}_sd"], s[f"{e}_mean"] + s[f"{e}_sd"], color=COL[e], alpha=0.18, lw=0)
            a.axhline(g, color="grey", ls="--", lw=1, zorder=0)
            if i == 0: a.set_title(f"$\\gamma={g}$", fontsize=10)
            if j == 0: a.set_ylabel(FIG[lab], fontsize=9)
            a.tick_params(labelsize=8); a.set_ylim(-0.065, 0.065)
    for j in range(3): ax[2, j].set_xlabel("Supplementary / original sample size", fontsize=9)
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 0.98]); fig.savefig(p("fig_main_simulation_band.pdf"), bbox_inches="tight"); plt.close(fig)

def table1(m20, t):
    nA = int(m20.groupby(["label", "gamma"]).size().min()); 
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Rejection rate of $H_0:\gamma=0$ at the 5\% level (supplementary sample $=$ 20\% of the original; " + f"{nA} replications per cell). "
         + r"\emph{Heckman} uses the corrected two-step standard errors of \citet{Heckman79}; \emph{Proposed} is $\hat\gamma_Q$ with the standard error of Section~\ref{sec:correction}. "
         + r"Panel B repeats the $\gamma=0$ case with $\varepsilon_i$ drawn from a Student-$t$ distribution rescaled to the same variance.}",
         r"\label{tab:reject}", r"\begin{tabular}{llccc}", r"\toprule", r"Selection preference & $\gamma$ / Errors & None & Heckman & Proposed \\", r"\midrule",
         r"\multicolumn{5}{l}{\emph{Panel A: Normal errors}} \\"]
    for lab in LAB:
        for k, g in enumerate(GAM):
            x = m20[(m20.label == lab) & np.isclose(m20.gamma, g)]
            gs = f"$-{abs(g):.2f}$" if g < 0 else f"$\\phantom{{-}}{g:.2f}$"
            L.append(f"{LONG[lab] if k == 0 else '':26s} & {gs} & {rej(x.none_p):.3f} & {rej(x.heck_p):.3f} & {rej(x.qw_p):.3f} \\\\")
    L += [r"\addlinespace", r"\multicolumn{5}{l}{\emph{Panel B: fat-tailed errors, $\gamma=0$}} \\"]
    for lab in LAB:
        for k, e in enumerate(["t10", "t5", "t3"]):
            x = t[(t.label == lab) & (t.eps == e)]
            L.append(f"{LONG[lab] if k == 0 else '':26s} & $t({e[1:]})$ & {rej(x.none_p):.3f} & {rej(x.heck_p):.3f} & {rej(x.qw_p):.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(p("tab_reject.tex"), "w").write("\n".join(L) + "\n")

def table2(S, has_mle, nmin):
    s = S[S.ratio > 0]
    head = (r"& \multicolumn{3}{c}{Proposed ($Q$ known)} & \multicolumn{2}{c}{Likelihood ($Q$ unknown)} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-6}" + "\n"
            + r"Supplement & Max $|$bias$|$ & SE/SD & Coverage & Coverage (Wald) & Coverage (LR) \\") if has_mle else \
           r"Supplement & Max $|$bias$|$ & SE/SD & Coverage \\"
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Bias, standard errors, and 95\% confidence-interval coverage, summarized over the nine designs of Figure~\ref{fig:main} (at least " + f"{nmin}"
         + r" replications per design and supplementary-sample size). SE/SD is the average standard error divided by the Monte Carlo standard deviation of $\hat\gamma_Q$; coverage is the average over designs, with the minimum in parentheses"
         + (r"; LR intervals for the likelihood estimator of Section~\ref{sec:mle} invert its likelihood-ratio test.}" if has_mle else ".}"),
         r"\label{tab:coverage}", r"\begin{tabular}{lccccc}" if has_mle else r"\begin{tabular}{lccc}", r"\toprule", head, r"\midrule"]
    for r, x in s.groupby("ratio"):
        row = f"{int(round(r*100))}\\% & {np.abs(x.qw_mean - x.gamma).max():.4f} & {(x.qw_se_mean / x.qw_sd).mean():.2f} & {x.qw_cov.mean():.3f} ({x.qw_cov.min():.3f})"
        if has_mle: row += f" & {x.mle_cov.mean():.3f} ({x.mle_cov.min():.3f}) & {x.mle_cov_lr.mean():.3f} ({x.mle_cov_lr.min():.3f})"
        L.append(row + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(p("tab_coverage.tex"), "w").write("\n".join(L) + "\n")

def table3(full):
    pr = full["probit"]; star = lambda q: "^{***}" if q < 0.01 else ("^{**}" if q < 0.05 else ("^{*}" if q < 0.10 else ""))
    lab = {"AR_pos": r"Positive-return part, $\hat b_1$", "AR_neg": r"Negative-return part, $\hat b_2$", "Female": "Female",
           "logMV": r"$\log$(market value)", "Outsider": "Outsider appointee", "Analyst": r"$\log(1+\text{analyst following})$"}
    L = [r"\begin{table}[t]", r"\centering", r"\caption{Coverage-equation Probit estimates (full sample, $n=" + f"{full['n']:,}".replace(",", "{,}") + r"$; standard errors in parentheses).}",
         r"\label{tab:realprobit}", r"\begin{tabular}{lcc}", r"\toprule", r"& Coefficient & \\", r"\midrule"]
    for k in ["AR_pos", "AR_neg", "Female", "logMV", "Outsider", "Analyst"]:
        c = pr[k]; st = f"${star(c['p'])}$" if star(c["p"]) else ""
        L.append(f"{lab[k]} & {c['coef']:.3f} & ({c['se']:.3f}){st} \\\\")
    L += [r"\bottomrule", r"\multicolumn{3}{l}{\footnotesize $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$} \\", r"\end{tabular}", r"\end{table}"]
    open(p("tab_realprobit.tex"), "w").write("\n".join(L) + "\n")

def figure2(R, full, suffix):
    for fn, cols in [(f"fig_realdata{suffix}.pdf", ["none", "heck", "qw"]), (f"fig_realdata_zoom{suffix}.pdf", ["none", "qw"])]:
        fig, a = plt.subplots(figsize=(3.3, 2.8)); bm = full["full_mean"]
        for e in cols:
            x = R.groupby("ratio")[f"{e}_est"].agg(["mean", "std"]).dropna(subset=["mean"]).reset_index()
            a.plot(x.ratio, x["mean"], marker=MK[e], color=COL[e], label=NM[e], ms=4, lw=1.3)
            a.fill_between(x.ratio, x["mean"] - x["std"].fillna(0), x["mean"] + x["std"].fillna(0), color=COL[e], alpha=0.15, lw=0)
        a.axhline(bm, color="grey", ls=":", lw=1.2, zorder=0); a.text(0.92, bm, "full-sample\nraw mean", fontsize=7.5, color="grey", va="center")
        a.set_xlabel("Supplementary / WSJ-covered sample", fontsize=9); a.set_ylabel("Estimated avg. abnormal return (%)", fontsize=9)
        a.tick_params(labelsize=8); a.legend(fontsize=7.5, loc="upper right"); fig.tight_layout(); fig.savefig(p(fn), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    GAM = [-0.02, 0.0, 0.02]; N = ["# Numbers quoted in the paper, by section", ""]
    simf = p("sim_reps_quick.csv" if a.quick else "sim_reps.csv")
    if os.path.exists(simf):
        d = pd.read_csv(simf); m, t = d[d.design == "main"], d[d.design == "tdist"]
        has_mle = "mle_est" in d and d.mle_est.notna().any()
        S = summarize(m); S.to_csv(p("sim_summary.csv"), index=False)
        m20 = m[np.isclose(m.ratio, 0.20)]; nmin = int(m[m.ratio > 0].groupby(["label", "gamma", "ratio"]).size().min())
        figure1(S); table1(m20, t); table2(S, has_mle, nmin)
        at = lambda lab, g, r: S[(S.label == lab) & np.isclose(S.gamma, g) & np.isclose(S.ratio, r)].iloc[0]
        mb = lambda r, e: max(abs(at(l, g, r)[f"{e}_mean"] - g) for l in LAB for g in GAM)
        ratios = sorted(r for r in S.ratio.unique() if r > 0); asym = LAB[:2]
        R0 = {(l, c): rej(m20[(m20.label == l) & np.isclose(m20.gamma, 0.0)][c]) for l in LAB for c in ["none_p", "heck_p", "qw_p", "mle_p", "mle_lr_p_true"]}
        N += [f"Replications: {int(m[np.isclose(m.ratio,0.05)].groupby(['label','gamma']).size().min())} per design and supplement size, {int(m20.groupby(['label','gamma']).size().min())} at 20% (Sec. 3, Fig. 1 caption)", "",
              "## Section 2.6",
              "Complement-pool (supplement-only) mean, prefers positive, gamma=0.02, supplements 5-50%: " + ", ".join(f"{at('Prefers positive',0.02,r).comp_mean:+.4f}" for r in ratios),
"## Section 3",
              "Heckman average estimate, prefers positive, gamma=0, at 5/20/50%: " + ", ".join(f"{at('Prefers positive',0.0,r).heck_mean:+.4f}" for r in [0.05, 0.2, 0.5]),
              f"Max |bias| at 5%: None {mb(0.05,'none'):.4f}, Heckman {mb(0.05,'heck'):.4f}, Proposed {mb(0.05,'qw'):.4f}; Proposed max over all supplements {max(mb(r,'qw') for r in ratios):.4f}",
              f"Rejection at gamma=0, 20% (asymmetric designs): None {pct(min(R0[(l,'none_p')] for l in asym))}-{pct(max(R0[(l,'none_p')] for l in asym))}, Heckman {pct(min(R0[(l,'heck_p')] for l in asym))}-{pct(max(R0[(l,'heck_p')] for l in asym))}",
              f"Rejection at gamma=0, 20% (all designs): Proposed {pct(min(R0[(l,'qw_p')] for l in LAB))}-{pct(max(R0[(l,'qw_p')] for l in LAB))}",
              "Power at gamma=+-0.02, 20%: Proposed min " + pct(min(rej(m20[(m20.label == l) & np.isclose(m20.gamma, g)].qw_p) for l in LAB for g in [-0.02, 0.02])),
              "Panel B, Proposed: " + "; ".join(f"{e} {pct(min(rej(t[(t.label==l)&(t.eps==e)].qw_p) for l in LAB))}-{pct(max(rej(t[(t.label==l)&(t.eps==e)].qw_p) for l in LAB))}" for e in ["t10", "t5", "t3"]),
              "Panel B, max over designs: None " + pct(max(rej(t[(t.label == l) & (t.eps == e)].none_p) for l in LAB for e in ["t10", "t5", "t3"])) + ", Heckman " + pct(max(rej(t[(t.label == l) & (t.eps == e)].heck_p) for l in LAB for e in ["t10", "t5", "t3"])),
              "Table 2 summary, Proposed: SE/SD " + f"{min((S[np.isclose(S.ratio,r)].qw_se_mean/S[np.isclose(S.ratio,r)].qw_sd).mean() for r in ratios):.2f}-{max((S[np.isclose(S.ratio,r)].qw_se_mean/S[np.isclose(S.ratio,r)].qw_sd).mean() for r in ratios):.2f}"
              + f"; mean coverage {min(S[np.isclose(S.ratio,r)].qw_cov.mean() for r in ratios):.3f}-{max(S[np.isclose(S.ratio,r)].qw_cov.mean() for r in ratios):.3f}; min {S[S.ratio>0].qw_cov.min():.3f}"]
        if has_mle:
            N += [f"Likelihood estimator: max |bias| at 5/20/50%: {mb(0.05,'mle'):.4f}, {mb(0.2,'mle'):.4f}, {mb(0.5,'mle'):.4f}",
                  f"Likelihood estimator, rejection at gamma=0, 20%: Wald {pct(min(R0[(l,'mle_p')] for l in LAB))}-{pct(max(R0[(l,'mle_p')] for l in LAB))}, LR {pct(min(R0[(l,'mle_lr_p_true')] for l in LAB))}-{pct(max(R0[(l,'mle_lr_p_true')] for l in LAB))}",
                  "Likelihood estimator, Panel B: Wald " + f"{pct(min(rej(t[(t.label==l)&(t.eps==e)].mle_p) for l in LAB for e in ['t10','t5','t3']))}-{pct(max(rej(t[(t.label==l)&(t.eps==e)].mle_p) for l in LAB for e in ['t10','t5','t3']))}"
                  + ", LR " + f"{pct(min(rej(t[(t.label==l)&(t.eps==e)].mle_lr_p_true) for l in LAB for e in ['t10','t5','t3']))}-{pct(max(rej(t[(t.label==l)&(t.eps==e)].mle_lr_p_true) for l in LAB for e in ['t10','t5','t3']))}",
                  f"Likelihood estimator, mean coverage Wald {min(S[np.isclose(S.ratio,r)].mle_cov.mean() for r in ratios):.3f}-{max(S[np.isclose(S.ratio,r)].mle_cov.mean() for r in ratios):.3f}, LR {min(S[np.isclose(S.ratio,r)].mle_cov_lr.mean() for r in ratios):.3f}-{max(S[np.isclose(S.ratio,r)].mle_cov_lr.mean() for r in ratios):.3f}"]
    for tag in ["raw", "winsor"]:
        rf, jf = p(f"realdata_reps_{tag}.csv"), p(f"realdata_full_{tag}.json")
        if not (os.path.exists(rf) and os.path.exists(jf)): continue
        R, F = pd.read_csv(rf), json.load(open(jf))
        G = R.groupby("ratio").agg(none=("none_est", "mean"), heck=("heck_est", "mean"), qw=("qw_est", "mean"), qw_sd=("qw_est", "std"),
                                   mle=("mle_est", "mean"), mle_se=("mle_se", "median")).reset_index()
        G.to_csv(p(f"realdata_summary_{tag}.csv"), index=False)
        figure2(R, F, "" if tag == "raw" else "_winsor")
        if tag == "raw": table3(F)
        g = lambda c, r: G.loc[np.isclose(G.ratio, r), c].values[0]
        N += ["", f"## Section 4 [{tag}]",
              f"Benchmark (full-sample mean) {F['full_mean']:+.3f} (SE {F['full_se']:.3f}); median {F['full_median']:+.3f}; covered mean {F['covered_mean']:+.3f}; Q = {F['Q']:.3f}; n = {F['n']}, covered = {F['n_covered']}",
              f"Probit (Table 3): b1 {F['probit']['AR_pos']['coef']:.3f} (SE {F['probit']['AR_pos']['se']:.3f}), b2 {F['probit']['AR_neg']['coef']:.3f} (p = {F['probit']['AR_neg']['p']:.2f})",
              f"None: 0% {g('none',0.0):+.3f}, 20% {g('none',0.2):+.3f}, 90% {g('none',0.9):+.3f}",
              f"Heckman: range {G.heck.min():+.3f} to {G.heck.max():+.3f}; 90% {g('heck',0.9):+.3f}",
              "Proposed (Q-weighted mean), average: " + ", ".join(f"{int(r*100)}% {g('qw',r):+.3f}" for r in [0.05, 0.2, 0.5, 0.9]),
              "Proposed analytic SE: " + ", ".join(f"{int(r*100)}% {F['qw_se_by_ratio'][str(r)]:.3f}" for r in [0.05, 0.2, 0.5]),
              f"Proposed SD across draws at 5%: {g('qw_sd',0.05):.3f}"]
        if "mle_full" in F:
            N.append("Likelihood estimator: " + ", ".join(f"{int(r*100)}% {g('mle',r):+.3f}" for r in [0.05, 0.2, 0.5, 0.9]) + f"; median SE at 5% {g('mle_se',0.05):.3f}; all events {F['mle_full']:+.3f} (SE {F['mle_full_se']:.3f})")
    open(p("numbers_for_text.md"), "w").write("\n".join(N) + "\n")
    print("\n".join(N))
