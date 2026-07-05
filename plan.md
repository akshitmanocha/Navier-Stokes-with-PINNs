# Project Plan — Epidemic-PINN: Recovering a Time-Varying Transmission Rate β(t) and R₀(t) from Real COVID-19 Data

> **One-line pitch:** A physics-informed neural network that fits a country's real COVID-19 case curve with an SEIR model and *discovers the hidden, time-varying transmission rate β(t)* — revealing how the reproduction number R₀(t) rose and fell around lockdowns — then is rigorously stress-tested against noise, data sparsity, and a classical baseline.

This document is the blueprint. It is grounded in current facts (verified June 2026) about the dataset, the epidemiology, and PINN engineering. Nothing below is guessed — sources are cited in §15.

---

## 1. Why this is a 10/10 portfolio project

The single framing — **inverse problem on real data** — hits all four things you asked for at once:

| Flavor | How this project delivers it |
|---|---|
| **Cool visual** | The headline figure: fitted epidemic curve + a recovered **R₀(t) timeline annotated with real lockdown dates**, overlaid against an independent Rₜ estimate. Immediately legible to a non-expert. |
| **Rigorous benchmark** | Recovered R₀(t) vs. OWID's own `reproduction_rate`; error vs. **noise level**; error vs. **data sparsity**; vs. a **classical least-squares baseline**; ablations; all with **error bars over seeds**. |
| **Real problem** | Real, downloadable COVID-19 data (Our World in Data). Tangible stakes: estimating R₀(t) is exactly what public-health teams did to evaluate interventions. |
| **Novel method** | The twist that makes vanilla PINNs work here: **β represented as a small neural network of time, β_φ(t)**, with a smoothness prior and positivity constraint — instead of a single fixed constant. |

**Narrative arc for the repo (great for recruiters):** *"I started from the canonical Raissi Navier-Stokes PINN, understood the physics-informed loss, then applied the same idea to a real-world inverse problem in epidemiology — recovering an unobservable, time-varying physical quantity from noisy public data, with a rigorous evaluation."* The fluids work becomes a "methods validation" prequel.

---

## 2. Background: the model and the inverse problem

### 2.1 The SEIR compartmental model

Population `N` split into **S**usceptible, **E**xposed (infected, not yet infectious), **I**nfectious, **R**ecovered. The ODE system:

```
dS/dt = − β(t) · S · I / N
dE/dt = + β(t) · S · I / N − σ · E
dI/dt = + σ · E − γ · I
dR/dt = + γ · I
```

- **β(t)** — transmission rate (per day). *This is the unknown we recover.* It changes over time as behavior/policy change.
- **σ** — rate E→I = 1 / latent period. **Fixed** (see §5.3).
- **γ** — rate I→R = 1 / infectious period. **Fixed** (see §5.3).

(We'll also implement the simpler **SIR** model — no E compartment — as a sanity-check baseline and ablation.)

### 2.2 Key formulas (the quantities we report)

- **Basic reproduction number:** `R₀(t) = β(t) / γ`
- **Effective reproduction number:** `R_eff(t) = (β(t) / γ) · S(t) / N` — this is the directly interpretable one (crosses 1.0 when the epidemic turns over) and is what we compare against OWID's `reproduction_rate`.
- **Observation operator** (links the unobservable model to reported data): reported daily new cases ≈ `ρ · σ · E(t)` (new symptomatic/ascertained cases flow out of E at rate σ), where **ρ** is the reporting/ascertainment fraction. For SIR: new cases ≈ `ρ · β(t) · S · I / N`.

### 2.3 Why this is an *inverse* problem (and why PINNs shine)

We **only observe a noisy, scaled proxy of one compartment** (reported new cases). We must recover the full latent state trajectory **and** a free function β(t). Classical methods (e.g., EpiEstim/renewal equation) estimate Rₜ but assume a fixed generation-interval distribution and don't enforce the full mechanistic ODE. A PINN simultaneously (a) fits the data, (b) enforces the SEIR ODE as a soft constraint, and (c) infers β(t) — all via autograd. This "recover a hidden physical quantity consistent with both data and dynamics" is exactly the PINN sweet spot.

---

## 3. Data

### 3.1 Source and the (helpful) fact that it's frozen

**Our World in Data (OWID) COVID-19 dataset.** Verified June 2026:
- The dataset **stopped updating 19 Aug 2024**; the GitHub repo `owid/covid-19-data` was **archived 24 Mar 2026**.
- **This is a feature, not a bug, for a portfolio project:** the data is now a *fixed, citable snapshot* → perfect reproducibility forever. We will **download once and cache locally** (commit a small per-country CSV slice, not the 100MB+ megafile) so the project runs offline and identically for any reviewer.

### 3.2 Download URLs (both still live as of June 2026; cache locally)

- Legacy megafile (frozen): `https://covid.ourworldindata.org/data/owid-covid-data.csv`
- GitHub raw mirror: `https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv`
- Codebook: `https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-codebook.csv`
- New catalog (compact): `https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv`

**Loader strategy:** try local cache → else download legacy megafile → slice to chosen country/date range → save slice to `data/cache/<iso>.csv`. Be robust to either schema (legacy uses `location`/`iso_code`; compact uses `country`/`code`).

### 3.3 Columns we use

| Concept | Column |
|---|---|
| Daily new cases (raw) | `new_cases` |
| **New cases, 7-day smoothed** (use this) | `new_cases_smoothed` |
| New deaths, smoothed | `new_deaths_smoothed` |
| Population | `population` |
| **Oxford stringency index** (lockdown intensity, 0–100) | `stringency_index` |
| **Independent Rₜ estimate** (validation target) | `reproduction_rate` |
| Identifiers | `location`/`country`, `iso_code`/`code`, `date` |

### 3.4 Data-quality gotchas (must handle)

- **Use the smoothed series.** Raw daily counts are extremely spiky (weekend/weekly reporting, especially 2022+). `new_cases_smoothed` interpolates across reporting gaps.
- **Negative corrections → NaN.** When a source revised downward, OWID set the day to NaN. Don't naïvely forward-fill; handle NaNs explicitly.
- **`reproduction_rate` and `stringency_index` stop being populated before the case data ends** — clip the analysis window to where they exist for the validation/annotation figures.
- **Focus on a clean window.** Best signal: the **first 1–3 waves** of a country with clear policy timing (e.g., Germany, Italy, UK, South Korea) in 2020–2021, where reporting was daily and lockdowns are well-documented.

---

## 4. The novel twist (the contribution)

Vanilla PINN inverse problems learn a **single constant** unknown parameter (as in DINN, Shaier & Raissi 2021). Real epidemics have a **time-varying** transmission rate. Our contribution, following the Karniadakis-group / Millevoi line of work:

> **Represent β as a dedicated small neural network β_φ(t)** — a separate MLP that takes time and outputs the transmission rate — with (a) a **positivity/boundedness output transform**, and (b) a **smoothness regularizer** on dβ/dt. This turns an ill-posed free-function recovery into a stable, interpretable estimate.

We then **honestly evaluate identifiability** — the project's intellectual backbone (§5.3, §12): we show what β(t)'s *shape* can and cannot be recovered from cases alone, and how a constant vs. time-varying reporting rate ρ affects it (Gostic et al. 2020).

---

## 5. Method

### 5.1 Architecture (two-network design)

- **State network** `u_θ(τ) → [S, E, I, R] / N`: MLP, ~4–5 hidden layers × 32–64 neurons, tanh. Output through a softmax-like or sigmoid normalization so compartments are positive and ≈ sum to 1.
- **Parameter network** `β_φ(τ) → β(t)`: small MLP, ~2–3 layers × 16–32 neurons, tanh, with output transform (§5.4).
- Input is **scaled time** τ ∈ [0,1] (§5.2).

(Alternative single-network design — Long et al. 2021, one net with 7 outputs — will be mentioned and optionally implemented as an ablation.)

### 5.2 Normalization (the highest-leverage step — do first)

- **Time:** `τ = (t − t₀) / T`, `T = t_f − t₀`, so τ ∈ [0,1].
- **States:** divide by population `N` → all in [0,1].
- **Residual chain-rule factor:** because the net differentiates w.r.t. τ, every time derivative gains a `1/T` factor. The SIR residual becomes `r_s = (1/T)·ds/dτ + β·s·i = 0`. **Forgetting this 1/T factor is the #1 silent ODE-PINN bug** — bake it into the residual function and unit-test it.

### 5.3 Identifiability decisions — FIX σ and γ, LEARN β(t)

This is the most important modeling decision and a key talking point:
- **Fix σ and γ.** Trying to learn β, σ, γ simultaneously is non-identifiable (many combinations fit the same curve). Literature consensus (Kharazmi 2021; Cai/Karniadakis 2022) is to fix the latent and infectious rates from clinical estimates and learn only transmission.
- **Defaults (with a generation-interval consistency check):**
  - Infectious period 1/γ ≈ **5 days → γ = 0.2** (Millevoi et al. 2023).
  - Latent period 1/σ: clinical incubation ≈ 5.2 days → σ ≈ 0.19. **But** to keep the mean generation interval ≈ 5–6 days (since 1/σ + 1/γ ≈ GI), a shorter latent (~2–3 days, σ ≈ 0.33–0.5) is more defensible due to pre-symptomatic transmission (Tang et al. 2021). **We will treat σ, γ as fixed hyperparameters and run a sensitivity sweep** over the literature range — turning a modeling caveat into an experiment.
- **Reporting rate ρ:** fix to a constant (e.g., scale so the model's `σ·E` peak matches the observed-case peak), and **explicitly discuss** that a *constant* ρ does not bias the *shape* of R₀(t), while a *time-varying* ρ does (Gostic et al. 2020). This is stated as a limitation, not hidden.

### 5.4 β(t) parameterization

- **Positivity / boundedness:** `β(t) = β_min + (β_max − β_min)·sigmoid(g_φ(t))` (or `softplus`). Prevents negative/exploding transmission. *(Millevoi et al. found a **hard** positivity constraint via the output activation more robust than a soft negative-output penalty — we use the bounded-sigmoid form.)*
- **Smoothness prior — two complementary mechanisms:** (a) keep β_φ **low-capacity** (1–2 layers × 5–16 neurons), which is itself a smoothness prior (Cai/Karniadakis use 1×5); (b) add `λ_smooth · mean[(dβ/dt)²]` (first-derivative Tikhonov). Tune λ_smooth by an L-curve sweep — the largest value that doesn't degrade the data fit.
- **Optional functional ansatz (regularization sweep):** instead of a free net, β(t) can be a constrained form with few learnable coefficients — piecewise-constant per-week, or a sigmoid/exponential "lockdown" shape (Olumoyin et al. 2021). Worth a one-line ablation: free-net vs. low-capacity-net vs. ansatz.

### 5.5 Loss

```
L = λ_data · L_data            (MSE: model-predicted reported cases vs observed new_cases_smoothed)
  + λ_res  · L_residual        (MSE of the 4 SEIR ODE residuals at collocation times)
  + λ_ic   · L_ic              (initial-condition match)
  + λ_smooth · L_smooth        (smoothness of β(t))
```

- **Adaptive weighting** between data and residual: gradient-norm / learning-rate annealing (Wang et al. 2021), EMA α=0.9, update every ~10–100 steps. `stop_gradient` on the weights. Start with fixed weights, escalate to adaptive only if needed.
- **`L_ic` matters:** Millevoi et al. add an explicit initial-condition loss (squared error on S(0), I(0), R(0)) rather than folding ICs into the data term — recommended, since ICs anchor the otherwise-underdetermined latent trajectory.
- **Optional "split" training (Millevoi et al., ~1 order-of-magnitude accuracy gain):** Stage 1 — fit a smooth differentiable surrogate to the case data only (plain regression). Stage 2 — freeze that surrogate and train the physics (ODE residual + IC + β_φ) against it. Listed as a P1 enhancement if the joint loss is unstable.

### 5.6 Optimization (two-stage, standard recipe)

1. **Adam** — lr 1e-3 with exponential decay, ~10k–50k iterations (global search).
2. **L-BFGS** — `strong_wolfe` line search, polish to a tight minimum (hundreds–few-thousand iters).

### 5.7 Device & precision strategy

- ODE PINNs are **tiny** → train in **seconds–minutes**. CPU is fine.
- **float64 matters for ODE-PINN accuracy and L-BFGS.** Apple **MPS does not support float64**. Strategy:
  - **M4 Pro:** run **CPU + float64** for the small models (fast enough; numerically clean). Use MPS+float32 only if profiling shows a real speedup.
  - **H100:** float64 fully supported; use it only for the big experiment grids (multi-country × multi-seed × sweeps), well within the 9-hour cap.

---

## 6. Experiments (what earns "research-grade")

Every experiment reports **mean ± std over ≥5 random seeds**.

1. **Headline — β(t)/R₀(t) recovery.** Fit one clean country (e.g., Germany 2020–2021). Plot recovered R_eff(t) vs OWID `reproduction_rate`; annotate lockdown dates from `stringency_index`. Report fit error (relative L2 on cases) and agreement with the independent Rₜ.
2. **Noise robustness.** Add 1%, 5%, 10%, 20% Gaussian (and Poisson/Rician-style) noise to cases; measure degradation of recovered R₀(t). (Mirrors Raissi-style Table 2 robustness.)
3. **Data sparsity.** Subsample observation days (every 1/3/7/14 days); find where recovery breaks. Real-world relevant (weekly reporting).
4. **Multiple countries.** Run 3–5 countries with different policy timing; show the method generalizes and the R₀(t) drops align with each country's interventions.
5. **Baseline comparison.** (a) Classical least-squares / sliding-window fit of constant β; (b) renewal-equation Rₜ (EpiEstim-style). Show the PINN's time-varying β(t) tracks reality better than piecewise-constant fits, and is ODE-consistent unlike EpiEstim.
6. **Ablations.** physics loss on/off; fixed vs adaptive loss weights; smoothness λ sweep; SIR vs SEIR; σ/γ sensitivity sweep.
7. **(P1) Uncertainty quantification.** Deep ensemble (5–10 seeds) → confidence band on R₀(t) + calibration commentary.

---

## 7. Figures (the deliverable visuals)

1. **Hero figure:** R_eff(t) recovered vs OWID Rₜ, with lockdown shading + the fitted case curve (twin axis). *(one great PNG/GIF for the README top)*
2. Noise-robustness curve (recovery error vs noise %), with error bars.
3. Sparsity curve (recovery error vs observation interval).
4. Multi-country small-multiples of R₀(t) with policy annotations.
5. PINN vs baseline overlay.
6. Ablation bar chart (physics on/off, weighting, SIR/SEIR).
7. (P1) Ensemble uncertainty band on R₀(t).
8. Loss curves + ODE-residual sanity plot (shows the physics is actually satisfied).

---

## 8. Repository architecture (replaces the 6 loose scripts)

```
epidemic-pinn/
├── README.md                  # story + hero figure + how-to-reproduce
├── plan.md                    # this file
├── report/
│   └── report.md (or .tex)    # paper-style writeup: abstract→method→results→limitations→refs
├── requirements.txt / environment.yml
├── pyproject.toml             # installable package
├── configs/                   # YAML configs — every run fully specified
│   ├── base.yaml
│   ├── germany_seir.yaml
│   ├── noise_sweep.yaml
│   └── sparsity_sweep.yaml
├── src/epidemic_pinn/
│   ├── data/                  # OWID download/cache, country slicing, smoothing, NaN handling
│   ├── models/                # state net, beta net, SIR/SEIR wrappers, baselines
│   ├── physics/               # SEIR residuals (with 1/T factor), R0/Reff formulas, obs operator
│   ├── train/                 # config-driven loop, Adam→LBFGS, adaptive weights, seeding
│   ├── eval/                  # metrics (rel-L2), sweeps, comparison to OWID Rt, baselines
│   └── viz/                   # all publication figures
├── experiments/               # one script per figure/table; writes results CSV
├── results/                   # metrics CSVs + generated figures (committed)
├── tests/                     # pytest: residual 1/T factor, shapes, seeding, R0 formula
└── .github/workflows/ci.yml   # lint + tests on push
```

Config-driven, **seeded**, logged (CSV/JSON; optional Weights & Biases), **tested**, **CI'd**, one-command reproduction.

---

## 9. Two-week schedule (~2h/day active + background training)

> Background-friendly: experiments are tiny and fast, so "background training" = kick off the multi-country × multi-seed × sweep grid (optionally on H100) while you do other things.

### Week 1 — foundation & first real result
- **Day 1 — Setup & repo skeleton.** Clean Python 3.11/3.12 venv (system Python is bare — no torch/scipy); install deps; scaffold package; init git; CI stub. Fix the inherited bugs as a warm-up commit (the `print` outside the loop in `Train.py`; `weights/` missing).
- **Day 2 — Data pipeline.** OWID loader + local cache + country slicing + smoothing + NaN handling. Sanity plots of cases/`reproduction_rate`/`stringency_index`. Commit a small cached slice.
- **Day 3 — Forward SEIR + physics module.** Implement SEIR residuals with the **1/T factor**, R₀/R_eff formulas, observation operator. Unit-test the residual against a classical ODE solver (scipy `solve_ivp`) on known β. *(This test is gold — proves the physics is right.)*
- **Day 4 — PINN v1 on synthetic data.** Generate synthetic SEIR data with a *known* β(t), then verify the PINN **recovers the known β(t)**. This is the controlled ground-truth validation — do it before touching real data.
- **Day 5 — PINN on real data (headline).** Fit Germany (or chosen country). Produce the first hero figure: recovered R_eff(t) vs OWID Rₜ + lockdown shading. Debug normalization/weighting.
- **Day 6 — Stabilize & config-ify.** Move everything to YAML configs; add seeding, CSV logging, Adam→LBFGS; tune smoothness λ. Re-run headline cleanly.
- **Day 7 — Buffer / baseline #1.** Implement classical least-squares baseline; first comparison plot. Catch-up day.

### Week 2 — rigor, breadth, polish
- **Day 8 — Noise robustness experiment** + figure (error bars over seeds).
- **Day 9 — Data sparsity experiment** + figure.
- **Day 10 — Multi-country runs** (3–5 countries) + small-multiples figure. Kick off as a background grid.
- **Day 11 — Ablations** (physics on/off, fixed vs adaptive weights, SIR vs SEIR, σ/γ sensitivity) + figure.
- **Day 12 — (P1) UQ deep ensemble** + uncertainty-band figure. (Skip to polish if behind.)
- **Day 13 — Report writing.** Paper-style `report.md`/`.tex`: abstract, method, results tables, **limitations** (identifiability, time-varying ρ), references. Polish all figures.
- **Day 14 — Final polish.** README with hero figure + reproduction instructions; clean tests/CI green; record a short demo (GIF or notebook); final commit + tag.

**Priority tiers** (so a slip never kills the project):
- **P0 (must ship):** Days 1–9, 13–14 → data pipeline, validated PINN, headline result, noise + sparsity sweeps, baseline, report, polished repo. *This alone is a strong portfolio piece.*
- **P1 (strong stretch):** multi-country, ablations, UQ.
- **P2 (bonus):** single-network ablation, fractional-SEIR, an interactive Gradio/Streamlit demo.

---

## 10. Compute mapping

| Task | Where | Why |
|---|---|---|
| All dev, synthetic validation, single-country fits | **M4 Pro, CPU + float64** | Tiny models, seconds–minutes; float64 clean (MPS lacks it) |
| Big grid: multi-country × ≥5 seeds × noise/sparsity sweeps | **H100** (optional) | Embarrassingly parallel; finishes far under the 9-hour cap |

No GPU is strictly required — H100 just makes the final sweep grid fast. The M4 Pro handles everything for P0.

---

## 11. Deliverables checklist (what makes it 10/10)

- [ ] Real, cached, reproducible dataset + robust loader
- [ ] Validated physics (residual unit-tested vs scipy ODE solver)
- [ ] **Controlled recovery of a *known* β(t) on synthetic data** (proves correctness)
- [ ] **Hero figure:** recovered R₀(t)/R_eff(t) vs independent Rₜ + lockdown annotations
- [ ] Noise-robustness + sparsity sweeps, **error bars over seeds**
- [ ] Classical baseline comparison (physics-informed beats it / is more consistent)
- [ ] Ablations + σ/γ sensitivity
- [ ] (P1) Uncertainty quantification
- [ ] Config-driven, seeded, tested, CI-green, one-command reproduction
- [ ] Paper-style report **with an honest limitations section**
- [ ] Clean README telling the fluids→epidemiology story

---

## 12. Honest limitations to state in the report (this *raises* credibility)

- **Identifiability:** the *level* of β(t)/R₀(t) is confounded with the reporting fraction ρ; we recover the *shape/temporal trend* robustly, the absolute level conditionally on fixed ρ, σ, γ.
- **Time-varying reporting biases Rₜ** (Gostic et al. 2020): testing ramp-ups can masquerade as rising transmission. We discuss and, where possible, restrict to windows of stable reporting.
- **Fixed σ, γ** are modeling choices; we quantify sensitivity rather than claim a single truth.
- **Frozen, possibly revised data**; OWID set negative corrections to NaN.
- **SEIR is a simplification** (no age structure, waning immunity, variants). The method is the contribution, not a definitive epidemiological estimate.

Stating these *demonstrates research maturity* — exactly what a reviewer/recruiter looks for.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| PINN won't converge / trivial fit | Normalize first; synthetic-data validation (Day 4) before real data; adaptive weighting; Adam→LBFGS |
| β(t) overfits noise / goes negative | Smoothness prior + bounded output transform |
| Data URLs go down | **Cache locally and commit the slice** (data is frozen anyway) |
| MPS float64 errors | Use CPU+float64 on Mac for the tiny models |
| Scope creep | Strict P0/P1/P2 tiers; P0 is self-sufficient |

---

## 14. Open questions for you (defaults chosen if you don't answer)

1. **Country for the headline?** *Default: Germany* (clean daily reporting, well-documented lockdown dates, populated `reproduction_rate`). Alternatives: Italy, UK, South Korea.
2. **Report format?** *Default: Markdown* `report.md` (renders on GitHub) — or LaTeX if you want a true paper PDF.
3. **Experiment tracking?** *Default: local CSV/JSON* (zero setup, offline) — or Weights & Biases for nicer plots.
4. **SEIR vs SIR as primary?** *Default: SEIR primary, SIR as ablation.*

---

## 15. References (verified during planning)

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. — the PINN foundation (your existing fluids project).
- Kharazmi, Cai, Zheng, Lin, Karniadakis (2021). *Identifiability and predictability of integer- and fractional-order epidemiological models using PINNs.* Nature Computational Science. — **anchor**; fixes σ, learns time-varying β; identifiability conclusions.
- Shaier, Raissi, Seshaiyer (2021). *Disease-Informed Neural Networks (DINN).* arXiv:2110.05445 — learns parameters as constants; contrast case.
- Millevoi, Pasetto, et al. (2023/24). *A PINN approach for epidemiological inverse problems (SIR/β(t)).* arXiv:2311.09944 — separate β(t) network, positivity, loss decomposition; γ=0.2 (5-day infectious).
- Long, Khaliq, Furati (2021). *Identification and prediction of time-varying parameters of COVID-19 via PINNs.* arXiv:2103.09949 — single net, time-varying β/γ/µ, sigmoid bounding.
- Cai, Karniadakis et al. (2022). fractional SEIR PINN, arXiv:2205.11379 — separate small β(t)/µ(t) networks; σ fixed.
- Gostic et al. (2020). *Practical considerations for measuring Rₜ.* PLoS Comput. Biol. — constant vs time-varying reporting and Rₜ bias. **The identifiability reference.**
- Wang, Teng, Perdikaris (2021). *Understanding and mitigating gradient pathologies in PINNs.* — learning-rate-annealing loss weighting (α=0.9).
- Wang et al. (2023). *An Expert's Guide to Training PINNs.* — non-dimensionalization, grad-norm weighting, Adam→LBFGS recipe.
- Lauer et al. (2020), Annals Int. Med. — COVID incubation ~5.1 days (σ).
- Tang et al. (2021), Front. Public Health — generation-interval consistency; shorter effective latent period.
- Our World in Data COVID-19 dataset (frozen 2024-08-19; repo archived 2026-03-24). https://github.com/owid/covid-19-data
- Oxford COVID-19 Government Response Tracker (Blavatnik School) — `stringency_index`.
- Arroyo Marioli et al. (2020), TrackingR — OWID's `reproduction_rate` method.
