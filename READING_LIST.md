# Reading List — Epidemic-PINN Project

A prioritized, verified reading list before/while building the project. **You do not need to read all of these deeply.** Follow the tiers:

- **🔴 Deep-read (5 papers)** — the core; understand these thoroughly.
- **🟡 Skim (5 papers)** — extract one specific idea each.
- **⚪ Reference (5 papers)** — know they exist, cite them, dip in only if needed.

Total realistic effort: ~2 focused days of reading, spread across week 1 while you set up the repo.

> All links verified to exist (June 2026). arXiv links are open-access; DOI links may be paywalled (use arXiv where given).

---

## 🔴 TIER 1 — DEEP-READ (the core five)

### 1. Raissi, Perdikaris & Karniadakis (2019) — *Physics-Informed Neural Networks* ⭐ THE foundation
- **J. Comput. Phys.** | arXiv: https://arxiv.org/abs/1711.10561 (Part I) & https://arxiv.org/abs/1711.10566 (Part II)
- Difficulty: medium | ~3h
- **Why:** Defines the PINN and, crucially, the **inverse problem** (learn unknown equation coefficients jointly with the solution) — exactly our structure.
- **Extract:** (1) autograd residual idea; (2) inverse formulation — promote β, γ to trainable; (3) composite loss = data term + ODE-residual term; (4) the **Adam→L-BFGS** optimizer recipe everyone copies.

### 2. Kharazmi, Cai, … Karniadakis (2021) — *Identifiability & predictability of epidemic models with PINNs* ⭐ THE anchor
- **Nature Computational Science** | https://doi.org/10.1038/s43588-021-00158-0
- Difficulty: hard | ~2h
- **Why:** Directly asks *can you even recover β(t)/R₀(t) from limited COVID case data?* — the identifiability question that defines what your project can honestly claim.
- **Extract:** separate sub-network per time-varying parameter; data-mismatch + ODE-residual loss; **which parameters MUST be fixed vs learned**, and the warning that fitting cases alone leaves β(t) non-identifiable unless constrained. This sets your fix-σ-and-γ decision.

### 3. Olumoyin, Khaliq & Furati (2021) — *EINN: time-varying infection rates of COVID-19* ⭐ closest blueprint
- arXiv: https://arxiv.org/abs/2104.02603 (also *Infectious Disease Modelling*)
- Difficulty: medium | ~1.5h
- **Why:** The cleanest worked example of *your exact task* — learn β(t) from real case data, track R₀(t) under mitigation, validated on Italy/South Korea/UK/US (your countries!).
- **Extract:** β(t) as a smooth NN output of time; composite loss on cumulative + daily cases + ODE residual; which params they fix vs learn; how R₀(t) is computed post-hoc from β(t).

### 4. Wang, Teng & Perdikaris (2021) — *Understanding & mitigating gradient pathologies in PINNs* ⭐ makes it converge
- SIAM J. Sci. Comput. | arXiv: https://arxiv.org/abs/2001.04536
- Difficulty: medium | ~2h
- **Why:** Diagnoses the **#1 reason a PINN inverse problem stalls** — unbalanced gradients between data and residual loss — and gives the fix.
- **Extract:** the **learning-rate-annealing loss-weighting rule** (λ̂ᵢ = max|∇L_r| / mean|∇Lᵢ|, EMA update). This is the adaptive weighting in plan.md §5.5.

### 5. Cori, Ferguson, Fraser & Cauchemez (2013) — *EpiEstim: time-varying reproduction numbers* ⭐ your baseline
- **Am. J. Epidemiol.** | https://doi.org/10.1093/aje/kwt133
- Difficulty: medium | ~2h
- **Why:** THE standard classical Rₜ estimator. You **benchmark the PINN against this** — and OWID's own `reproduction_rate` uses a related method.
- **Extract:** renewal-equation estimator Rₜ = Iₜ / Σ I_{t−s} wₛ; Bayesian Gamma/Poisson sliding-window posterior; how the serial interval enters. Use the `epyestim` Python package to produce baseline curves with credible intervals.

---

## 🟡 TIER 2 — SKIM (one idea each)

### 6. Karniadakis et al. (2021) — *Physics-informed machine learning* (review)
- **Nature Reviews Physics** | https://doi.org/10.1038/s42254-021-00314-5 | easy | ~2h
- **Extract:** vocabulary/taxonomy (observational/inductive/learning biases), the framing of inverse problems & "discovering hidden physics", and the catalog of training pathologies + mitigations. Great for your report's intro.

### 7. Gostic et al. (2020) — *Practical considerations for measuring Rₜ* ⭐ honesty reference
- **PLOS Comput. Biol.** | https://doi.org/10.1371/journal.pcbi.1008409 | medium
- **Extract:** the failure modes (reporting delays, smoothing artifacts, infection-vs-onset, time-varying reporting bias) your PINN must avoid, **and their synthetic-data validation protocol** — adopt it to score your PINN. This justifies your limitations section.

### 8. Shaier, Raissi & Seshaiyer (2022) — *DINN: Disease-Informed Neural Networks*
- arXiv: https://arxiv.org/abs/2110.05445 | easy | ~30 min
- **Extract:** the generic, accessible architecture for learning epidemic parameters as **constants**. Build this first as your **constant-β baseline**, then extend β → β(t). Good warm-up.

### 9. Wang, Sankaran, Wang & Perdikaris (2023) — *An Expert's Guide to Training PINNs*
- arXiv: https://arxiv.org/abs/2308.08468 | medium | skim the recipe/tables
- **Extract:** the end-to-end checklist — **non-dimensionalize first**, Fourier features, grad-norm/NTK weighting with EMA, curriculum/causal time-weighting, Adam + exp-decay schedule. Your go-to when training misbehaves.

### 10. Millevoi, Pasetto & Ferronato (2024) — *PINN approach for compartmental models*
- **PLOS Comput. Biol.** | arXiv: https://arxiv.org/abs/2311.09944 | medium
- **Extract:** the **"reduced-split" training** (fit data first, then ODE residual → ~1 order-of-magnitude accuracy gain), hard positivity via output activation, and model reduction. A P1 enhancement if joint training is unstable.

---

## ⚪ TIER 3 — REFERENCE (cite / dip in if needed)

### 11. Hethcote (2000) — *The Mathematics of Infectious Diseases*
- **SIAM Review** | https://doi.org/10.1137/S0036144500371907
- **Extract:** rigorous SIR/SEIR ODE derivations, R₀ = β/γ, threshold theorems, standard parameter ranges. The reference for setting up the physics correctly.

### 12. Lu, Meng, Mao & Karniadakis (2021) — *DeepXDE library*
- **SIAM Review** | arXiv: https://arxiv.org/abs/1907.04502
- **Extract:** standard PINN API + sane defaults (Glorot init, tanh, Adam→L-BFGS), and Residual-based Adaptive Refinement (RAR). Optional implementation harness — though we're building from scratch for portfolio clarity.

### 13. Long, Khaliq & Furati (2021) — *Time-varying params of COVID-19, recover-then-forecast*
- arXiv: https://arxiv.org/abs/2103.09949
- **Extract:** the verification trick (re-integrate learned params through a standard ODE solver to check consistency) and the LSTM-forecasts-β(t) extension. Template for the **P2 forecasting bonus**.

### 14. Li et al. (2020) — *Substantial undocumented infection of SARS-CoV-2*
- **Science** | https://doi.org/10.1126/science.abb3221
- **Extract:** ~86% of early infections undocumented → justifies the **reporting/ascertainment factor ρ** in your observation model. Use to set a ρ prior.

### 15. Lagaris, Likas & Fotiadis (1998) — *Neural nets for solving ODEs/PDEs*
- **IEEE Trans. Neural Networks** | https://doi.org/10.1109/72.712178
- **Extract:** the PINN precursor; the trial-solution trick to **hard-enforce initial conditions** (I(0)=I₀). Historical context + one optional stabilization technique.

---

## Suggested reading order (mapped to the build)

| When | Read | Goal |
|---|---|---|
| **Before coding** | #1 Raissi, #6 review (skim) | Understand PINNs + inverse problems |
| **Before SEIR module** | #11 Hethcote (skim), #2 Kharazmi | Get the physics + identifiability right |
| **Before PINN v1** | #3 Olumoyin, #8 DINN | Concrete β(t) blueprint + constant baseline |
| **When training stalls** | #4 Wang gradients, #9 Expert's Guide | Fix convergence |
| **Before baseline expt** | #5 Cori/EpiEstim, #7 Gostic | Build + validate the comparison |
| **For the report** | #6, #7, #2 | Framing, limitations, honesty |

**The non-negotiable three** if you're short on time: **#1 (Raissi)**, **#2 (Kharazmi)**, **#3 (Olumoyin)**. Everything else is support.
