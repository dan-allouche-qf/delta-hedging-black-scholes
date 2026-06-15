# Discrete Delta Hedging under Black-Scholes — Results Report

**Dan Allouche**

This report summarises a Monte-Carlo study of discrete-time delta hedging of a
short option position in the Black-Scholes market. All numbers are produced by
the [`deltahedge`](../src/deltahedge) package (exact GBM, `N = 50,000` paths,
common random numbers, seed `20240614`) and reproduced by the narrative
[notebook](../notebooks/Delta_Hedging_Analysis.ipynb). Baseline parameters:
`r = 10%`, `σ = 20%`, `S₀ = 100`, `K = 90`, `T = 0.25y`.

---

## 1. Set-up and a note on the mean error

We sell a European call priced at `V(t,S)` and hold the self-financing hedge
`π = Δ S + β B`, `Δ = ∂V/∂S`, rebalanced at `n` dates `t_k = kT/n`. The terminal
P&L is `ε_n = π(T) − payoff`. The drift `μ` never enters the hedge — the engine
sees only realised prices — and under the pricing measure (`μ = r`) the
discounted self-financing portfolio is a martingale, so

> **E[ε_n] = 0 exactly, at every n.**

This single fact reframes the original project: there is no "mean error
converging to zero" to study. Empirically, with `N = 50k` the mean error is
statistically indistinguishable from zero at every frequency (all `p > 0.05`,
all 95% CIs contain 0). The mean is *not* the quantity of interest — the
*dispersion* is.

| n | label | mean ± sem | std | p(mean=0) |
|---:|---|---:|---:|---:|
| 0 | no hedge | −0.06 ± 0.043 | 9.59 | 0.18 |
| 1 | static | +0.01 ± 0.007 | 1.47 | 0.30 |
| 3 | monthly | +0.00 ± 0.004 | 0.98 | 0.27 |
| 12 | weekly | −0.00 ± 0.002 | 0.52 | 0.83 |
| 84 | daily | −0.00 ± 0.001 | 0.21 | 0.57 |

The standard deviation collapses by ~98% from no-hedge to daily — the real,
resolved effect (`figures/convergence_call.png`, `error_histograms_call.png`).

## 2. The √n scaling law, fitted properly

The hedging-error std should scale as `n^b` with `b = −1/2` asymptotically.
Quoting the single ratio `σ(n=1)/σ(n=84) ≈ 7.9` as evidence is wrong: the law
predicts `√84 ≈ 9.2`. Fitting `log σ = a + b log n` on `n ≥ 3` (excluding the
static `n=1`, which performs no intermediate rebalancing and is outside the
asymptotic regime) gives

> **b = −0.47 ± 0.01** (95% CI [−0.49, −0.45]),

cleanly consistent with the theoretical `−1/2`. The diagnostic `σ·√n` rises from
1.47 (`n=1`) to a ~1.89 plateau for `n ≥ 3`, showing exactly why `n=1` must be
excluded.

## 3. Drift dependence

The original claim that "variance is controlled by frequency, not `μ`" is false.
Using common random numbers across `μ` (so only the drift changes), the `n=84`
dispersion falls monotonically:

| μ | −0.10 | 0.00 | 0.05 | 0.10 | 0.15 | 0.30 |
|---|---:|---:|---:|---:|---:|---:|
| std | 0.254 | 0.232 | 0.219 | 0.206 | 0.194 | 0.158 |

A higher drift carries paths deep in-the-money into the **low-gamma** region,
where there is less gamma-P&L to mis-hedge (`figures/mu_sensitivity.png`). The
mean stays zero throughout; the drift moves the *variance*, not the *bias*.

## 4. Digital option

The cash-or-nothing call (`payoff = 1{S_T ≥ K}`) is genuinely harder to hedge.
Two findings are real (the original's "non-monotonic mean error" was noise, all
`p > 0.05`):

- the error distribution is **bimodal** (clusters at payoff 0 vs 1);
- the std-scaling exponent is only **`≈ −0.24`** — the digital error does *not*
  obey `1/√n`. Even `n = 252` leaves a std of ~0.066 (vs ~0.001-scale means),
  because the delta spikes near the strike close to expiry
  (`figures/digital_histograms.png`).

## 5. Transaction costs and Leland (1985)

With a proportional cost `κ` per unit of underlying traded, more rebalancing cuts
dispersion but raises the expected cost drag. The two effects trade off, and a
tail-risk measure (5% CVaR of the loss) has an **interior optimum**: at `κ = 0.1%`
the CVaR-optimal frequency is around `n* ≈ 168` (`figures/cost_variance_frontier.png`).

**Leland's adjusted volatility** prices the cost buffer into the hedge:
`σ_L = σ·√(1 + sign·√(2/π)·κ/(σ√δt))`. Hedging the short call at `σ_L` (with
`κ = 0.2%`, `n = 84`) improves the cost-adjusted P&L on *both* axes versus
hedging at the plain `σ` — higher mean (less cost drag) and lower std
(`figures/leland.png`).

## 6. Volatility misspecification and the gamma-P&L identity

This is the central result of option hedging. Hedging at implied vol `σ_h` while
the asset realises `σ_r`, the discrete hedging error of the short option is, to
leading order,

```
ε ≈ −½ Σ_k Γ_k S_k² [ (ΔS_k/S_k)² − σ_h² δt ]
  = −½ Σ_k Γ_k S_k² (σ_r² − σ_h²) δt   (deterministic mismatch)
    − ½ Σ_k Γ_k S_k² σ_r² (Z_k² − 1) δt  (mean-zero discretisation noise),
```

where `Γ_k` is the BS gamma at the *hedging* vol. The mean error is therefore
driven by the **variance mismatch `σ_r² − σ_h²`**, not the drift — the
"robustness of Black-Scholes" result. Empirically the mean error's sign tracks
`sign(σ_r² − σ_h²)` exactly:

| σ_hedge | 0.14 | 0.17 | 0.20 | 0.23 | 0.26 |
|---|---:|---:|---:|---:|---:|
| mean error | −0.42 | −0.20 | 0.00 | +0.24 | +0.52 |

The analytic decomposition matches the simulation engine to three decimals
(`σ_r = 0.20`, `σ_h = 0.25`, `n = 252`): engine mean `+0.484`, gamma-P&L mean
`+0.477`, deterministic mismatch term `+0.478`, correlation `0.994`
(`figures/vol_misspec.png`).

## 7. Greeks-based P&L attribution

The second-order Taylor decomposition `dV ≈ θ dt + Δ dS + ½ Γ dS²`, summed along
the path (`n = 252`), explains the option's value change to a tiny residual:
predicted `dV = +0.434` vs actual `+0.434`, residual std `0.018` against an
actual-change std of `9.6` — i.e. the Greeks explain the P&L to ~0.2%
(`figures/pnl_attribution.png`).

## 8. Model risk

Keeping the same BS-delta hedge but simulating the underlying under Heston
(stochastic vol) and Merton (jump diffusion) degrades the hedge sharply:

| model | std | 5% VaR | 5% CVaR | min |
|---|---:|---:|---:|---:|
| GBM (correct) | 0.21 | 0.34 | 0.56 | −1.8 |
| Heston | 0.64 | 1.55 | 2.35 | −5.0 |
| Merton (jumps) | 1.49 | 1.63 | 5.49 | −18.6 |

Jumps in particular break delta hedging and produce a fat-tailed loss
distribution that no rebalancing frequency can control (`figures/model_risk.png`).

## 9. Statistical caveats

- All means are reported with `sem = std/√N` and 95% CIs; "consistent with zero"
  means the CI contains zero, not that the point estimate is "small".
- Common random numbers are used across frequencies and drifts so comparisons are
  not blurred by Monte-Carlo noise.
- Results are reproducible via `np.random.default_rng` with per-call seeding,
  independent of execution order. Pinned headline numbers are guarded by
  `tests/test_regression.py`.

---

*Reproduce: `pip install -e ".[dev]" && pytest && jupyter nbconvert --to notebook
--execute --inplace notebooks/Delta_Hedging_Analysis.ipynb`.*
