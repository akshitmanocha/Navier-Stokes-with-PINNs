# Recovering a Time-Varying Transmission Rate from COVID-19 Cases with a Physics-Informed Neural Network

## Abstract

The transmission rate β(t) of an epidemic — and the reproduction number R₀(t) = β(t)/γ derived from it — is the quantity public-health teams most want to track, yet it is never observed directly. We recover it as an **inverse problem**: given only a noisy, scaled proxy of one compartment (reported daily cases), reconstruct the hidden, time-varying β(t) that, fed through a mechanistic SEIR model, reproduces the data. We represent β(t) as a small neural network and train it by backpropagating a case-fit loss **through a differentiable SEIR ODE integrator**, so the epidemiological dynamics hold exactly by construction. On synthetic data with a known β(t) the method recovers it to **2.6 % relative-L2 error** (identifiable window, 5 % noise), and degrades gracefully to 3.8 % at 20 % noise and 4.0 % with only biweekly observations. On Germany's first COVID-19 wave it recovers an effective reproduction number R_eff(t) that matches Our World in Data's *independent* estimate to **RMSE 0.14**, placing the epidemic's turning point (R_eff = 1) on **8 April 2020**, about two weeks after the national lockdown. It beats a classical rolling-window growth-rate estimator by ~10× on the synthetic benchmark. We are explicit about the method's central limitation: β(t) is identifiable only where infectious prevalence is non-negligible, and its absolute level is confounded with the (fixed) reporting fraction.

## 1. Problem

The SEIR model splits a population of size *N* into **S**usceptible, **E**xposed, **I**nfectious, **R**ecovered fractions evolving as

```
dS/dt = − β(t) S I
dE/dt = + β(t) S I − σ E
dI/dt = + σ E − γ I
dR/dt = + γ I
```

β(t) is the transmission rate we want to recover; σ (incubation) and γ (recovery) are fixed. We observe only reported new cases, modelled as `ρ σ E(t)·N` — a scaled proxy for the flux out of E, where ρ is the reporting fraction.

This is an inverse problem with a **free-function** unknown. Classical PINN inverse problems recover a single constant; real epidemics need a *trajectory* β(t) that rises and falls with behaviour and policy. That is the contribution here.

## 2. Method

**β as a network.** β(t) = β_min + (β_max − β_min)·σ(g_φ(t)), where g_φ is a small (2×16 tanh) MLP of scaled time τ ∈ [0,1]. The sigmoid is a *hard* positivity/boundedness constraint; the small capacity is itself a smoothness prior, reinforced by a light penalty on (dβ/dτ)².

**Physics by construction.** Rather than enforce the ODE as a soft residual penalty on a separate state network, we **integrate the SEIR system with a differentiable RK4 scheme** (`src/seir.py::integrate_seir_torch`) driven by β_φ, and read modelled cases off the resulting E(t). Because the trajectory is a true solution of the ODE, conservation, positivity, and the S–E–I–R coupling are exact — there is no data-vs-residual weight to tune. The only unknowns are the β-network and the two initial infected fractions (E₀, I₀), each bounded so S₀ = 1 − E₀ − I₀ stays positive.

This is still a physics-informed neural network in spirit — a neural transmission rate trained through a mechanistic simulator — and proved far more robust on real low-attack-rate data than a soft-residual collocation PINN (see §5).

**Loss and optimisation.** A scale-free relative-MSE case-fit term plus the β-smoothness penalty, minimised with Adam (global search) then L-BFGS (polish). Everything runs CPU + float64 (Apple MPS lacks float64); a single fit takes well under a minute.

**Fixed σ, γ.** Learning β, σ, γ jointly is non-identifiable. We fix 1/γ = 5 days (γ = 0.2) and a short effective latent period 1/σ ≈ 3 days (σ = 0.33), consistent with pre-symptomatic transmission.

## 3. Results

### 3.1 Controlled validation (synthetic, known β)

We generate data from a known β(t) that drops sharply (a "lockdown", R₀ 2.5 → 1.1) during an active wave, add 5 % multiplicative noise, and recover β from cases alone.

| Quantity | Relative-L2 error |
|---|---|
| Recovered β(t), identifiable window | **0.026** |
| Recovered β(t), full window | 0.043 |
| Case fit | 0.009 |

Recovery is near-exact through the entire transition; the only deviation is in the early window where infectious prevalence is ≈0 and β is genuinely unidentifiable (§4). *Figure:* `figures/synthetic_recovery.png`.

### 3.2 Headline: Germany's first wave (real data)

Fit to Germany's reported cases, 1 Mar – 15 Aug 2020 (OWID, 7-day smoothed), ρ = 0.1.

| Quantity | Value |
|---|---|
| Case fit, relative-L2 | 0.121 |
| **R_eff(t) vs OWID `reproduction_rate`, RMSE** | **0.144** |
| Recovered R₀ range | [0.61, 3.32] |
| R_eff first crosses 1.0 | **8 April 2020** |

The recovered R_eff(t) — derived purely from the case curve — tracks OWID's completely independent reproduction-rate estimate closely: both start ≈3, fall through 1 in early April as lockdown intensity peaks, bottom out ≈0.6 in May, and drift back toward 1 over summer. The turning point lands ~2 weeks after Germany's 22 March lockdown, matching the incubation-plus-reporting lag. The recovered level is insensitive to ρ (RMSE 0.138–0.158 for ρ ∈ [0.05, 0.20]). *Figure:* `figures/headline_germany.png` (the hero figure).

### 3.3 Robustness (synthetic, multi-seed)

Mean ± std over 5 seeds, error on the identifiable window.

**Noise** (β-recovery error vs observation noise):

| Noise | 0 % | 2 % | 5 % | 10 % | 20 % |
|---|---|---|---|---|---|
| β rel-L2 | 0.027 | 0.026 | 0.030 | 0.034 | 0.038 |

**Sparsity** (5 % noise, β-recovery error vs reporting interval):

| Interval | 1 day | 3 days | 7 days | 14 days |
|---|---|---|---|---|
| # obs | 200 | 67 | 29 | 15 |
| β rel-L2 | 0.030 | 0.031 | 0.039 | 0.040 |

Degradation is graceful and monotone in both — the signature of a well-posed, stable inverse method. Notably, recovery barely worsens with biweekly data: the mechanistic constraint compensates for sparse observations. *Figures:* `figures/noise_sweep.png`, `figures/sparsity_sweep.png`.

### 3.4 Baseline comparison

Against a classical rolling-window estimator that infers the local exponential growth rate r(t) and maps it to R₀ via the SEIR relation R₀ = (1 + r/σ)(1 + r/γ):

| Method | R₀(t) rel-L2 (identifiable window) |
|---|---|
| Classical growth-rate | 0.250 |
| **PINN** | **0.027** |

The growth-rate baseline is both noisier and systematically biased (it ignores susceptible depletion and is not ODE-consistent), settling near 0.65 where the truth is 1.1. The PINN, enforcing the full mechanism, is ~10× more accurate. *Figure:* `figures/baseline_comparison.png`.

## 4. Limitations (stated plainly)

- **Identifiability.** β(t) is recoverable only where infectious prevalence I(t) is non-negligible; where I ≈ 0 the term β·S·I vanishes and cases carry no information about β. We always report error on this identifiable window and shade the rest in every figure.
- **Level vs shape / reporting.** With a *constant* reporting fraction ρ, the *shape and timing* of R₀(t) are robust and the level is only weakly sensitive to ρ (verified). A *time-varying* ρ (e.g. testing ramp-ups) would bias the level and could masquerade as changing transmission.
- **Fixed σ, γ** are modelling choices, not measurements; a fuller study would sweep them.
- **SEIR is a simplification** — no age structure, waning immunity, or variants. The method is the contribution, not a definitive epidemiological estimate for Germany.

Stating these is deliberate: the method's value is a *consistent, mechanistic, uncertainty-aware* reconstruction, not a claim of ground truth.

## 5. Notes on what worked

A soft-residual collocation PINN (separate state network + ODE-residual loss) was implemented first and **failed on real low-attack-rate data**: with S ≈ 1 and I ≈ 10⁻⁴, the state network's unobserved compartments drifted and the data-vs-residual weighting never stabilised (recovered R₀ collapsed to ≈0). Replacing it with the differentiable-integrator formulation fixed this at the root — physics holds exactly, there is nothing to weight, and recovery on a matched synthetic improved from ~0.45 to ~0.05 relative-L2. The lesson: for stiff, low-signal inverse ODE problems, integrating the dynamics beats penalising their residual.

## 6. Reproducing

```bash
python -m pytest -q          # physics unit tests (torch integrator == numpy oracle)
python run_all.py            # all experiments + figures (a few minutes, CPU)
python run_all.py --quick    # everything except the multi-seed sweeps
```

## References

Key sources: Raissi et al. (2019) PINNs; Kharazmi et al. (2021) identifiability of epidemiological PINNs; Millevoi et al. (2023) separate β(t) network and loss decomposition; Gostic et al. (2020) practical R_t and reporting bias; Wang et al. (2021, 2023) PINN training. Data: Our World in Data COVID-19 dataset (frozen 2024-08-19); Oxford COVID-19 Government Response Tracker (`stringency_index`); Arroyo Marioli et al. (2020) (`reproduction_rate`).
