# Efficient Monte Carlo Pricing of Barrier Options under the OUSV Model

Master's thesis on Monte Carlo pricing of path-dependent barrier options under the Ornstein-Uhlenbeck stochastic volatility model, with a focus on variance reduction and computational efficiency.

## Project Overview

The project studies the pricing of discretely monitored barrier options under the OUSV stochastic volatility model.

The analysis focuses on two contracts:

- up-and-in call options;
- up-and-out call options.

Barrier options are path-dependent derivatives, since their payoff depends not only on the terminal value of the underlying asset but also on whether a predefined barrier is reached during the life of the contract.

Monte Carlo simulation is therefore a natural pricing method.

The main objective is not only to obtain accurate prices, but to determine which estimator provides the best trade off between statistical precision and computational cost.

## Pricing Framework

Derivative prices are represented as discounted risk neutral expected payoffs.

When the expectation cannot be computed analytically, Monte Carlo simulation approximates it by generating simulated paths and averaging the corresponding discounted payoffs.

For barrier options, the full simulated path is required because the payoff depends on the barrier hitting event.

## Black-Scholes Benchmark

The Black Scholes model is introduced as a benchmark framework.

Its exact transition rule provides a simple environment for understanding Monte Carlo simulation before moving to stochastic volatility.

The limitations of constant volatility then motivate the use of the OUSV model.

## OUSV Model

The main pricing framework is the Ornstein Uhlenbeck stochastic volatility model.

The asset price is driven by a latent volatility factor following an Ornstein-Uhlenbeck mean reverting process.

Compared with Black-Scholes, the OUSV model allows volatility to evolve stochastically over time.

This additional flexibility makes the simulation problem more complex and requires numerical simulation schemes.

## Simulation Schemes

Three simulation schemes are compared.

### Euler Scheme

Euler provides the simplest numerical discretization and is used as the baseline.

It is computationally inexpensive but can introduce larger discretization bias.

### VHP Scheme

A scheme based on van Haastrecht, Lord and Pelsser improves on Euler by simulating the Ornstein-Uhlenbeck latent factor from its exact transition;

### Inverse Gaussian Scheme

The Inverse Gaussian Scheme provides a more refined approximation of the OUSV dynamics.

It simulates the integrated latent factor more accurately and approximates the conditional integrated variance through a moment matched Inverse Gaussian distribution.

The three schemes therefore represent different trade offs between computational cost and discretization accuracy.

## Fourier Pricing Benchmark

European vanilla call options under the OUSV model can be priced using a semi-closed Fourier pricing formula.

The Fourier price plays three roles in the project:

1. validation benchmark for the Monte Carlo simulation schemes;
2. known expectation for the control variate;
3. conditional continuation value in the Conditional Monte Carlo estimator.

The required Fourier integrals are evaluated numerically using Gauss-Laguerre quadrature.

## Monte Carlo Estimators

For each simulation scheme, four Monte Carlo estimators are compared.

### Plain Monte Carlo

Plain Monte Carlo directly averages the simulated discounted barrier-option payoffs.

It is used as the baseline estimator.

### Antithetic Variables

Antithetic sampling generates pairs of paths using opposite random shocks.

The two resulting payoffs are averaged in order to exploit negative dependence and potentially reduce estimator variance.

### Control Variates

The corresponding European vanilla call is used as the control variate.

The barrier-option payoff is adjusted using the difference between the simulated vanilla payoff and its known OUSV Fourier price.

The method is particularly effective when the barrier payoff is strongly correlated with the vanilla payoff.

An independent pilot sample is used to estimate the optimal control coefficient before the main Monte Carlo simulation.

### Conditional Monte Carlo

Conditional Monte Carlo replaces part of the remaining payoff uncertainty with a conditional expectation.

For an up-and-in option, once the barrier is reached, the remaining payoff becomes a vanilla call with the remaining maturity.

Instead of continuing the simulation to maturity, the residual vanilla call is priced using the OUSV Fourier formula.

## Numerical Design

The numerical analysis considers multiple barrier levels:

`B = {105, 110, 120, 130, 140, 150, 160, 190}`

Each barrier is treated as a separate contract.

For every combination of:

- option type;
- barrier level;
- simulation scheme;
- Monte Carlo estimator;

the analysis records:

- estimated price;
- standard error;
- end-to-end computational time;
- efficiency.

## Efficiency Criterion

Variance reduction alone is not sufficient to determine whether an estimator is practically useful.

Some techniques reduce the standard error but require substantially more computational time.

For this reason, the comparison uses the empirical efficiency indicator:

`Efficiency = Time × SE²`

where:

- `Time` is the total end-to-end computational time required to run the complete pricing procedure for a given barrier level, including random number generation, path simulation and estimator specific operations;
- `SE` is the estimated standard error of the Monte Carlo price.

Lower values indicate a better trade off between computational cost and statistical precision.

## Main Results

### Up-and-In Calls

Control Variates are generally the most efficient estimator for low, medium and moderately high barriers under all three simulation schemes.

For these barriers, the up-and-in payoff remains strongly related to the vanilla payoff, making the control variate particularly effective.

At the highest barrier, Plain Monte Carlo becomes competitive and is selected as the most efficient estimator.

Conditional Monte Carlo generally reduces the standard error but is penalized by the computational cost of repeated Fourier pricing evaluations at barrier hitting times.

### Up-and-Out Calls

The results are more heterogeneous.

For low barriers, Plain Monte Carlo and Antithetic Variables can remain competitive because many paths are knocked out early and the standard error is already small.

As the barrier increases, the up-and-out payoff becomes increasingly similar to the corresponding vanilla call payoff.

This strengthens the effectiveness of the Control Variate estimator.

## Comparison of Simulation Schemes

The Inverse Gaussian Scheme produces the smallest observed pricing deviation from the Fourier benchmark on both coarse and fine grids.

However, it is computationally more expensive.

On a sufficiently fine grid, Euler and VHP become statistically consistent with the Fourier benchmark while requiring substantially less computational time.

This highlights the trade off between simulation accuracy and computational cost.

## Main Takeaways

- Barrier option pricing under stochastic volatility requires full path simulation.
- The choice of simulation scheme affects both discretization bias and computational cost.
- Variance reduction is not automatically equivalent to higher computational efficiency.
- Control Variates are especially effective when the barrier payoff is strongly related to the vanilla payoff.
- Conditional Monte Carlo reduces variance but can become expensive because of repeated Fourier evaluations.
- The best estimator depends on the option type, barrier level and simulation scheme.

## Repository Structure

```text
efficient-monte-carlo-barrier-options/
│
├── README.md
└── efficient_monte_carlo_barrier_options_thesis.pdf
```

## Main File

- `efficient_monte_carlo_barrier_options_thesis.pdf` — complete Master's thesis including theoretical framework, OUSV simulation schemes, variance reduction methods, numerical experiments and results

## Technologies and Topics

- MATLAB
- Monte Carlo Simulation
- Computational Finance
- Derivatives Pricing
- Barrier Options
- Stochastic Volatility
- OUSV Model
- Variance Reduction
- Antithetic Variables
- Control Variates
- Conditional Monte Carlo
- Fourier Pricing
- Gauss-Laguerre Quadrature
- Numerical Methods

## Thesis Title

**Efficient Monte Carlo Pricing of Barrier Options under the OUSV Model: A Variance Reduction Approach**

Master's Degree in Finance  
Academic Year 2025–2026
