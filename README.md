# Efficient Monte Carlo Pricing of Barrier Options under the OUSV Model

**MSc thesis in Finance, University of Pavia (2025–2026)** — Supervisor: Prof. Riccardo Brignone · MATLAB

## Objective

The objective was to **compare Monte Carlo methods and simulation schemes for barrier option pricing under the Ornstein–Uhlenbeck stochastic volatility (OUSV) model**. The ultimate goal was to **identify the method offering the best trade-off between statistical precision and computational cost for different option types and barrier levels**.

Barrier options have no closed-form price under stochastic volatility, which is why they must be priced by Monte Carlo simulation.

## What was compared

| | Options considered |
|---|---|
| **Contracts** | Discretely monitored up-and-in and up-and-out calls, barriers from 105 to 190 (spot = strike = 100, 1-year maturity, 504 monitoring dates, 10,000 paths) |
| **Simulation schemes** | Euler · VHP (van Haastrecht, Lord and Pelsser) · Inverse Gaussian Scheme (Brignone and Sgarra) |
| **Estimators** | Plain Monte Carlo · Antithetic variables · Control variates · Conditional Monte Carlo |
| **Benchmark** | Semi-closed Fourier price of the European call (Gauss–Laguerre quadrature) |

Each method is ranked by an **efficiency indicator, Time × SE²**: a technique that reduces the standard error but takes much longer to run is not necessarily better.

## Key results

- **Control variates** (with the vanilla call as control) are the most efficient estimator for up-and-in calls at almost all barrier levels: at the lowest barrier they cut the standard error by up to **two orders of magnitude** compared with plain Monte Carlo.
- For **up-and-out calls** the answer depends on the barrier: plain Monte Carlo and antithetic variables win for low barriers (many paths are knocked out early), control variates for high barriers.
- **Conditional Monte Carlo** reduces variance but is penalised by the cost of repeated Fourier evaluations.
- **Simulation schemes:** against the Fourier benchmark, the Inverse Gaussian Scheme gives the smallest pricing error (about **12× smaller than Euler** on a coarse grid and **20× smaller** on a fine grid), but it is much slower. On a fine grid, Euler and VHP are already statistically consistent with the benchmark at a fraction of the cost.

**Takeaway:** there is no universally best method. The most efficient choice depends jointly on the option type, the barrier level and the simulation scheme.

## Repository

- `code/` — MATLAB implementation of the three simulation schemes, the four estimators, the OUSV Fourier pricing and the numerical experiments
- `efficient_monte_carlo_pricing_barrier_options_thesis.pdf` — full thesis (theory, simulation schemes, variance reduction methods, numerical results)
- `presentation.pdf` — thesis presentation

**Topics:** Monte Carlo simulation · stochastic volatility · barrier options · variance reduction · Fourier pricing · MATLAB
