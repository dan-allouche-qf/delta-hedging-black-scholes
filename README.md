# Delta Hedging in the Black-Scholes Framework

This project provides a comprehensive numerical analysis of the discrete-time delta hedging strategy for European Call Options within the Black-Scholes market model. It explores the relationship between rebalancing frequency and hedging error variance through Monte Carlo simulations.

## Problem Overview

In continuous-time finance, the Black-Scholes model suggests that an option can be perfectly replicated by a self-financing portfolio consisting of the underlying asset and a risk-free bond, provided that rebalancing occurs continuously. In practice, however, rebalancing only occurs at discrete intervals, leading to a residual risk known as the hedging error.

This study quantifies this error and demonstrates its convergence properties as the number of rebalancing steps increases.

## Features

- **Theoretical Rigor**: Includes the mathematical derivation of the self-financing property and the implementation of Black-Scholes pricing and Greeks (Delta and Gamma).
- **Simulation Engine**: A robust Monte Carlo framework to simulate asset price trajectories using Euler discretization.
- **Dynamic Rebalancing**: Implementation of discrete-time hedging logic that adjusts the portfolio at specified intervals.
- **Statistical Analysis**: Extensive evaluation of hedging error distributions, including mean, standard deviation, and tail risk (Min/Max).
- **Comparison of Frequencies**: Analysis across various regimes:
    - No Hedging
    - Single Hedging (at $t=0$)
    - Monthly Rebalancing
    - Weekly Rebalancing
    - Daily Rebalancing
- **Extension to Digital Options**: Comparative analysis showing the specific challenges of hedging discontinuous payoffs (Digital/Binary options).

## Mathematical Framework

The risky asset $S(t)$ follows a Geometric Brownian Motion:
$$dS(t) = S(t)\mu dt + S(t)\sigma dW(t)$$

The hedging portfolio $\pi(t)$ is maintained such that:
$$\pi(t) = \Delta(t)S(t) + \beta(t)B(t)$$

Where the error at maturity $T$ is defined as:
$$\varepsilon_n = \pi_n(T) - \max(S(T) - K, 0)$$

## Model Parameters

The analysis uses the following baseline configuration:

- **Risk-free Rate ($r$)**: 10.0%
- **Volatility ($\sigma$)**: 20.0%
- **Initial Asset Price ($S_0$)**: 100.00 EUR
- **Strike Price ($K$)**: 90.00 EUR
- **Maturity ($T$)**: 0.25 years (3 months)
- **Simulations**: 1,000 paths per frequency

## Project Structure

- `Delta_Hedging_Analysis.ipynb`: Detailed Jupyter Notebook containing theory, Python implementation, and result visualizations.
- `requirements.txt`: List of necessary Python libraries for reproduction.

## Installation

Ensure you have a Python environment (3.8+) ready. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Key Results

The simulations confirm that the standard deviation of the hedging error is inversely proportional to the square root of the rebalancing frequency ($\sqrt{n}$). While the mean error remains close to zero (under risk-neutral assumptions), the path-dependency of discrete hedging is clearly visualized through the narrowing of error histograms.

## Usage

To run the analysis and generate the plots:
1. Open the `Delta_Hedging_Analysis.ipynb` notebook in VS Code or Jupyter Lab.
2. Run all cells to execute the Monte Carlo simulations.
3. Observe the convergence behavior in the "Visualization" section.
