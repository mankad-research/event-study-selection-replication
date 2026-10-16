# Numbers quoted in the paper, by section

Replications: 400 per design and supplement size, 800 at 20% (Sec. 4, Fig. 1 caption)

## Section 3.2
Complement-pool (supplement-only) mean, prefers positive, gamma=0.02, supplements 5-50%: -0.0104, -0.0101, -0.0101, -0.0102, -0.0103, -0.0102, -0.0103
## Section 4
Heckman average estimate, prefers positive, gamma=0, at 5/20/50%: -0.0041, -0.0064, -0.0089
Max |bias| at 5%: None 0.0146, Heckman 0.0062, Proposed 0.0002; Proposed max over all supplements 0.0002
Rejection at gamma=0, 20% (asymmetric designs): None 91.5%-92.6%, Heckman 24.8%-28.5%
Rejection at gamma=0, 20% (all designs): Proposed 4.8%-7.0%
Power at gamma=+-0.02, 20%: Proposed min 100.0%
Panel B, Proposed: t10 4.5%-5.9%; t5 4.9%-5.1%; t3 5.6%-6.0%
Panel B, max over designs: None 91.5%, Heckman 22.4%
Table 2 summary, Proposed: SE/SD 0.98-1.01; mean coverage 0.945-0.956; min 0.927
Population coverage rate Q (normal errors), by design: Prefers positive g=-0.02: 0.542, Prefers positive g=+0.00: 0.572, Prefers positive g=+0.02: 0.643, Prefers negative g=-0.02: 0.643, Prefers negative g=+0.00: 0.572, Prefers negative g=+0.02: 0.542, Indifferent g=-0.02: 0.650, Indifferent g=+0.00: 0.626, Indifferent g=+0.02: 0.650
Likelihood estimator: max |bias| at 5/20/50%: 0.0002, 0.0001, 0.0002; max over all supplements 0.0003
Likelihood estimator fit diagnostics (all simulation fits): 36000 fits; no estimate 0; not converged 2; nu at lower bound 0; normal fallback 16086
Likelihood estimator, rejection at gamma=0, 20%: Wald 4.5%-7.0%, LR 4.4%-7.1%
Likelihood estimator, Panel B: Wald 4.4%-6.1%, LR 4.2%-5.6%
Likelihood estimator, mean coverage Wald 0.943-0.952, LR 0.944-0.953

## Section 5 [raw]
Benchmark (full-sample mean) -0.022 (SE 0.087); median -0.102; covered mean +0.193; Q = 0.507; n = 3103, covered = 1574
Probit (Table 3): b1 0.031 (SE 0.009), b2 0.004 (p = 0.55)
None: 0% +0.193, 20% +0.113, 90% -0.010
Heckman: range +0.189 to +0.261; 90% +0.189
Proposed (reweighted mean), average: 5% -0.040, 20% -0.044, 50% -0.024, 90% -0.019
Proposed analytic SE: 5% 0.270, 20% 0.146, 50% 0.105
Proposed SD across draws at 5%: 0.221
Outsider minus insider mean return: +0.762 (Welch t = 3.45)
WSJ article lag after press release (covered events): median 2 days; same day 16.8%; before the press release 1.0%; more than 30 days 0.6%
Likelihood estimator: 5% -0.151, 20% -0.095, 50% -0.045, 90% -0.026; median SE at 5% 0.086; all events -0.031 (SE 0.056, nu 2.03, converged True)
Likelihood estimator, supplement draws: 720 fits; no estimate 0; not converged 0; nu at lower bound 38; normal fallback 0; nu range 2.02-2.31, median 2.10

## Section 5 [winsor]
Benchmark (full-sample mean) +0.024 (SE 0.066); median -0.102; covered mean +0.163; Q = 0.507; n = 3103, covered = 1574
Probit (Table 3): b1 0.035 (SE 0.011), b2 0.012 (p = 0.28)
None: 0% +0.163, 20% +0.121, 90% +0.028
Heckman: range +0.185 to +0.236; 90% +0.189
Proposed (reweighted mean), average: 5% +0.062, 20% +0.038, 50% +0.034, 90% +0.023
Proposed analytic SE: 5% 0.205, 20% 0.111, 50% 0.079
Proposed SD across draws at 5%: 0.168
Outsider minus insider mean return: +0.724 (Welch t = 4.61)
WSJ article lag after press release (covered events): median 2 days; same day 16.8%; before the press release 1.0%; more than 30 days 0.6%
Likelihood estimator: 5% -0.147, 20% -0.075, 50% -0.020, 90% -0.022; median SE at 5% 0.100; all events -0.023 (SE 0.061, nu 2.95, converged True)
Likelihood estimator, supplement draws: 720 fits; no estimate 0; not converged 0; nu at lower bound 8; normal fallback 0; nu range 2.02-3.21, median 2.94
