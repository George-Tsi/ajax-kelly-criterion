"""
Kelly sizing for vertical credit spreads -- companion code to
"Kelly Criterion Position Sizing."

A binary bet has a closed-form Kelly fraction (f* = p - q/b). A credit
spread doesn't -- its payoff is piecewise-linear with a capped win, a
capped loss, and a continuum in between -- so f* has to be found
numerically: simulate terminal prices under GBM, price the spread's
payoff at each one, and grid-search for the f that maximizes E[ln(1+f*R)].

Run directly to reproduce the paper's GS $990/$1000 put spread example.

Author: George Tsiamtsiouris / AJAX Research
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KellyResult:
    f_star: float
    g_star: float
    f_grid: np.ndarray
    g_grid: np.ndarray
    half_kelly: float
    quarter_kelly: float


def simulate_terminal_prices(
    spot: float,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    n_simulations: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """GBM terminal prices: S_T = S0*exp[(r - sigma^2/2)T + sigma*sqrt(T)*Z].

    Drift is the risk-free rate, not a directional forecast -- keeps this
    isolated to the vol edge (subjective sigma vs implied) rather than
    smuggling in a price view.
    """
    if volatility < 0:
        raise ValueError(f"volatility must be non-negative, got {volatility}")
    if time_to_expiry <= 0:
        raise ValueError(f"time_to_expiry must be positive, got {time_to_expiry}")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")

    rng = np.random.default_rng(random_seed)
    z = rng.standard_normal(n_simulations)
    drift = (risk_free_rate - 0.5 * volatility**2) * time_to_expiry
    diffusion = volatility * np.sqrt(time_to_expiry) * z
    return spot * np.exp(drift + diffusion)


def spread_return(
    terminal_price: float | np.ndarray,
    short_strike: float,
    long_strike: float,
    credit: float,
) -> float | np.ndarray:
    """R(S) for a short put vertical: P&L(S) / max_risk, max_risk = width - credit.

    Three regions: capped win above the short strike, capped loss below
    the long strike, linear in between. Max risk is width minus the
    credit already collected -- not the width itself, which is just the
    strike distance, not capital actually at stake. (Dividing by width
    alone was a real bug in an earlier pass: it understates return per
    dollar at risk and inflates f* by width/max_risk, ~1.4x here.)

    Width isn't a separate arg -- it's short_strike - long_strike, always,
    so there's no way to pass one inconsistent with the strikes.

    All inputs must share one scale: per-share (credit=2.90), not total
    per-contract dollars (credit=290). R(S) is a ratio so either is fine
    on its own, but don't mix them in one call.
    """
    if long_strike >= short_strike:
        raise ValueError("long_strike must be below short_strike for a put credit spread")

    width = short_strike - long_strike
    if not 0 < credit < width:
        raise ValueError(
            f"credit ({credit}) must be strictly between 0 and width ({width}) -- "
            "check you passed per-share, not per-contract, dollars"
        )

    max_risk = width - credit
    pnl = np.where(
        terminal_price >= short_strike,
        credit,
        np.where(
            terminal_price <= long_strike,
            credit - width,
            credit - (short_strike - terminal_price),
        ),
    )
    return pnl / max_risk


def growth_rate(kelly_fraction: float, simulated_returns: np.ndarray) -> float:
    """Monte Carlo estimate of g(f) = E[ln(1 + f*R)] over the simulated returns."""
    return np.mean(np.log(1 + kelly_fraction * simulated_returns))


def solve_kelly_fraction(
    simulated_returns: np.ndarray,
    f_max: float = 1.0,
    n_grid_points: int = 200,
) -> KellyResult:
    """Grid-search f* = argmax g(f). g is concave, so the max is unique."""
    if f_max <= 0:
        raise ValueError(f"f_max must be positive, got {f_max}")
    if n_grid_points < 2:
        raise ValueError(f"n_grid_points must be at least 2, got {n_grid_points}")

    # Cap f_max so 1 + f*R can't go non-positive on the worst simulated
    # draw -- ln() of that would be NaN and silently wreck the grid search.
    worst_case_return = simulated_returns.min()
    if worst_case_return < 0:
        f_max = min(f_max, -0.999 / worst_case_return)

    f_grid = np.linspace(0.0, f_max, n_grid_points)
    g_grid = np.array([growth_rate(f, simulated_returns) for f in f_grid])
    best_idx = np.argmax(g_grid)
    f_star = f_grid[best_idx]

    return KellyResult(
        f_star=f_star,
        g_star=g_grid[best_idx],
        f_grid=f_grid,
        g_grid=g_grid,
        half_kelly=f_star / 2,
        quarter_kelly=f_star / 4,
    )


def simulate_price_paths(
    spot: float,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    n_paths: int,
    n_steps: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Same GBM model as simulate_terminal_prices, stepped n_steps times
    instead of jumped straight to T -- gives the full path, not just S_T,
    for the "spaghetti plot" chart. Shape (n_paths, n_steps + 1), column 0 = spot.
    """
    if volatility < 0:
        raise ValueError(f"volatility must be non-negative, got {volatility}")
    if time_to_expiry <= 0:
        raise ValueError(f"time_to_expiry must be positive, got {time_to_expiry}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    dt = time_to_expiry / n_steps
    rng = np.random.default_rng(random_seed)
    z = rng.standard_normal((n_paths, n_steps))
    step_log_returns = (risk_free_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z

    cumulative = np.cumsum(step_log_returns, axis=1)
    cumulative = np.hstack([np.zeros((n_paths, 1)), cumulative])
    return spot * np.exp(cumulative)


def run_kelly_case(
    spot: float,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    short_strike: float,
    long_strike: float,
    credit_per_share: float,
    n_simulations: int,
    label: str,
    expected_f_star: str,
) -> KellyResult:
    """Simulate -> price -> solve for one case, print a one-line result."""
    sim_prices = simulate_terminal_prices(
        spot, volatility, time_to_expiry, risk_free_rate, n_simulations, random_seed=42,
    )
    sim_returns = spread_return(sim_prices, short_strike, long_strike, credit_per_share)
    result = solve_kelly_fraction(sim_returns)
    print(f"[{label}] f* = {result.f_star:.3f}  (paper: {expected_f_star})")
    return result


def main() -> None:
    # GS $990/$1000 put spread, 2026-07-20 chain. Per-share units throughout
    # (credit=2.90, not the paper's total-contract $290) -- width comes from
    # the strikes inside spread_return, not passed in separately.
    #
    # n_simulations=5,000,000: at 100k draws this estimate is seed-dependent
    # noise -- across 10 seeds at 100k, f* ranged 0.077-0.084 (a real check
    # caught this, not a hunch). It settles at ~0.080 once the sample is
    # large enough, confirmed separately against a closed-form quadrature.
    common_args = dict(
        spot=1055.03,
        time_to_expiry=32 / 365,
        risk_free_rate=0.037,
        short_strike=1000.0,
        long_strike=990.0,
        credit_per_share=2.90,   # $19.875 - $16.975 mid-to-mid
        n_simulations=5_000_000,
    )

    run_kelly_case(volatility=0.350, label="Implied-vol case", expected_f_star="~0.000", **common_args)

    subjective_result = run_kelly_case(
        volatility=0.313, label="Subjective-vol case", expected_f_star="~0.080", **common_args,
    )
    print(f"  Half-Kelly = {subjective_result.half_kelly:.3f}   Quarter-Kelly = {subjective_result.quarter_kelly:.3f}")


if __name__ == "__main__":
    main()
