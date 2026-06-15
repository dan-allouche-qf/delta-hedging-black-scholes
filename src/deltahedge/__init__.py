"""deltahedge — discrete-time delta hedging in the Black-Scholes framework.

A small, tested library for studying how rebalancing frequency, drift,
transaction costs and volatility misspecification affect the hedging error of a
short option position, simulated by Monte-Carlo with the exact GBM scheme.
"""

from __future__ import annotations

from . import analysis, black_scholes, hedging, paths, plotting
from .analysis import fit_power_law, mc_summary, summary_table, var_cvar
from .hedging import (
    HedgeResult,
    gamma_pnl_decomposition,
    hedge_call,
    hedge_digital,
    hedge_error,
    leland_vol,
    pnl_attribution,
)
from .paths import crn_gbm_paths, gbm_paths, heston_paths, merton_paths

__version__ = "0.2.0"

__all__ = [
    "analysis",
    "black_scholes",
    "hedging",
    "paths",
    "plotting",
    "gbm_paths",
    "crn_gbm_paths",
    "heston_paths",
    "merton_paths",
    "hedge_error",
    "hedge_call",
    "hedge_digital",
    "gamma_pnl_decomposition",
    "leland_vol",
    "pnl_attribution",
    "HedgeResult",
    "mc_summary",
    "summary_table",
    "fit_power_law",
    "var_cvar",
]
