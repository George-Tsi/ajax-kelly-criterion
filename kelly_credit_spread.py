"""
Kelly Criterion Position Sizing for Vertical Credit Spreads
============================================================
Companion code to "Kelly Criterion Position Sizing" (Sections V-VIII).

Implements the numerical Kelly solver for a short (credit) vertical spread's
piecewise-linear payoff, since (per the paper) it has no closed-form f* the
way a binary bet does. Pure math/simulation -- no plotting dependency.

Usage:
    python kelly_credit_spread.py

Reproduces the paper's Section VII worked example (GS $990/$1000 put
spread) and prints the solved Kelly fractions.

Author: George Tsiamtsiouris / AJAX Research
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KellyResult:
    """Container for a solved Kelly optimization — paper Sections V, VIII."""

    f_star: float          # full-Kelly optimal fraction, argmax of g(f)
    g_star: float          # g(f*), the growth rate achieved at f_star
    f_grid: np.ndarray     # every candidate f tested
    g_grid: np.ndarray     # g(f) for every candidate — this is what gets charted
    half_kelly: float      # f_star / 2
    quarter_kelly: float   # f_star / 4


def simulate_terminal_prices(
    spot: float,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    n_simulations: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Draw simulated terminal underlying prices under GBM — paper Section V.

    S_T = S0 * exp[(r - sigma^2/2) * T + sigma * sqrt(T) * Z], Z ~ N(0, 1)

    Per the paper, drift is set to the risk-free rate (mu = r), not a
    directional forecast — this isolates the vol-selling edge (subjective
    sigma vs. implied sigma) rather than smuggling in a stock-picking view.

    Args:
        spot:            Current underlying price, S0.
        volatility:      Annualized volatility, sigma (e.g. 0.313 for 31.3%).
        time_to_expiry:  Time to expiration in years, T (e.g. 32/365).
        risk_free_rate:  Annualized risk-free rate, used as the drift, mu = r.
        n_simulations:   Number of Monte Carlo draws.
        random_seed:     Optional seed, for reproducible results.

    Returns:
        1D array of length n_simulations; each entry one simulated S_T.

    Raises:
        ValueError: If volatility is negative, time_to_expiry is not
            positive, or n_simulations is not positive.
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
    """
    Return on max-risk capital, R(S), for a short (credit) put vertical at
    expiration — paper Section V.

    Three regions, matching the paper exactly (width W = short_strike - long_strike):
      S >= short_strike:                    both legs worthless -> P&L = +credit (capped win)
      long_strike < S < short_strike:        short leg ITM, long leg OTM -> P&L = credit - (short_strike - S)
      S <= long_strike:                      both legs ITM -> P&L = credit - W (capped loss)

    R(S) = P&L(S) / (W - C), i.e. return on the max-risk capital tied up in
    the spread. Max risk is W - C (width minus the credit already collected),
    NOT the width itself -- W is the distance between strikes, not the amount
    of your own capital actually on the line. Dividing by W instead of (W - C)
    understates the return per dollar truly at risk, which biases the
    downstream Kelly fraction upward (this was a real bug in an earlier
    version: it inflated f* by a factor of W/(W-C), ~1.4x for the paper's
    GS worked example).

    Width is deliberately NOT a separate parameter: it's mathematically
    determined by the two strikes (W = short_strike - long_strike), never an
    independent quantity, so accepting it separately would let a caller pass
    a value inconsistent with the strikes -- computing it internally makes
    that class of error structurally impossible instead of just documented.

    Vectorized via np.where so this works identically whether terminal_price
    is a single float or a whole numpy array of simulated prices (the latter
    is what simulate_terminal_prices() produces).

    UNITS — all arguments must be on the SAME per-share scale (i.e. how
    options are actually quoted, e.g. "sold for $2.90"), not total per-contract
    dollars (which is credit/width * 100 shares). short_strike/long_strike/
    terminal_price are naturally per-share; credit must match that scale
    (e.g. credit=2.90 — NOT credit=290). R(S) is scale-invariant (it's a
    ratio), so per-share-consistent inputs give the identical, correct
    answer that dollar-consistent inputs would — the bug only appears if
    you mix the two scales in one call.

    Args:
        terminal_price: Underlying price(s) at expiration (S_T in the paper),
                         per-share.
        short_strike:   Strike of the short (sold) leg, K_s, per-share.
        long_strike:    Strike of the long (bought) leg, K_l, per-share. Must be < short_strike.
        credit:         Net credit received when opening the spread, C, per-share.
                         Must be strictly between 0 and the strike width.

    Returns:
        R(S): P&L(S) / (W - C), dimensionless return on max-risk capital.
        Same shape as terminal_price (scalar in, scalar out; array in, array out).

    Raises:
        ValueError: If long_strike is not below short_strike, or credit is
            not strictly between 0 and the strike width (an economically
            impossible credit spread -- e.g. max risk <= 0).
    """
    if long_strike >= short_strike:
        raise ValueError("long_strike must be below short_strike for a put credit spread")

    width = short_strike - long_strike
    if not 0 < credit < width:
        raise ValueError(
            f"credit ({credit}) must be strictly between 0 and the strike width ({width}) "
            "for a real credit spread — check you passed per-share, not total-contract, dollars"
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
    """
    Monte Carlo estimate of g(f) = E[ln(1 + f*R)] — paper Section V.

    Averages ln(1 + f*R) across every simulated outcome, approximating the
    true expectation since R(S) is piecewise-linear and has no closed-form
    integral against the GBM density.

    Deliberately takes the *already-computed* returns array rather than raw
    spread parameters: the optimizer calls this once per candidate f across
    a whole grid, and R(S) only needs to be computed once (via
    spread_return()) up front — recomputing it inside every call would redo
    the same work for every f.

    Args:
        kelly_fraction:    Candidate bet size f, as a fraction of bankroll.
        simulated_returns: Precomputed R(S) for every simulated terminal
                            price — i.e. spread_return() already applied to
                            simulate_terminal_prices()'s output.

    Returns:
        Estimated per-round compound growth rate, g(f), for this f.
    """
    return np.mean(np.log(1 + kelly_fraction * simulated_returns))


def solve_kelly_fraction(
    simulated_returns: np.ndarray,
    f_max: float = 1.0,
    n_grid_points: int = 200,
) -> KellyResult:
    """
    Numerically solve f* = argmax g(f) by grid search — paper Section V.

    Concavity (proved in Section II) guarantees a single, well-defined peak,
    so evaluating g(f) across an evenly-spaced grid and taking the best
    point is a safe way to locate it.

    Args:
        simulated_returns: Precomputed R(S) array (spread_return() already
                            applied to simulate_terminal_prices()'s output).
        f_max:              Upper bound of the search grid (fraction of bankroll).
        n_grid_points:       How many candidate f values to test.

    Returns:
        KellyResult with f_star, g_star, the full (f, g) grid for charting,
        and half/quarter-Kelly fractions (paper Section VIII).

    Raises:
        ValueError: If f_max is not positive or n_grid_points is below 2.
    """
    if f_max <= 0:
        raise ValueError(f"f_max must be positive, got {f_max}")
    if n_grid_points < 2:
        raise ValueError(f"n_grid_points must be at least 2, got {n_grid_points}")

    # Guard the search domain against the worst simulated outcome making
    # 1 + f*R non-positive, which would make ln(...) undefined (NaN) and
    # silently corrupt the grid search rather than raising. This is only
    # reachable for wide/thin-credit spreads or a caller-supplied f_max
    # pushed well above typical Kelly fractions -- harmless to always apply.
    worst_case_return = simulated_returns.min()
    if worst_case_return < 0:
        solvency_limit = -1.0 / worst_case_return
        f_max = min(f_max, solvency_limit * 0.999)

    f_grid = np.linspace(0.0, f_max, n_grid_points)
    g_grid = np.array([growth_rate(f, simulated_returns) for f in f_grid])

    best_idx = np.argmax(g_grid)
    f_star = f_grid[best_idx]
    g_star = g_grid[best_idx]

    return KellyResult(
        f_star=f_star,
        g_star=g_star,
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
    """
    Simulate full GBM price paths over time — not just the terminal price.

    Same model as simulate_terminal_prices (S_T = S0*exp[(r-sigma^2/2)T +
    sigma*sqrt(T)*Z]), but stepped forward n_steps times instead of jumping
    straight to T. This is what produces the classic Monte Carlo "spaghetti
    plot" of many fanning-out trajectories, since we keep every intermediate
    price rather than only the endpoint.

    Args:
        spot:            Current underlying price, S0.
        volatility:      Annualized volatility, sigma.
        time_to_expiry:  Total time horizon in years, T.
        risk_free_rate:  Annualized risk-free rate, used as drift, mu = r.
        n_paths:         How many separate simulated trajectories to draw.
        n_steps:         How many time steps to divide T into.
        random_seed:     Optional seed, for reproducible results.

    Returns:
        2D array, shape (n_paths, n_steps + 1). Each row is one full path;
        column 0 is spot (every path starts at the same known price today).

    Raises:
        ValueError: If volatility is negative, time_to_expiry is not
            positive, or n_paths/n_steps is not positive.
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

    cumulative_log_returns = np.cumsum(step_log_returns, axis=1)
    cumulative_log_returns = np.hstack([np.zeros((n_paths, 1)), cumulative_log_returns])

    return spot * np.exp(cumulative_log_returns)


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
    """
    Run one Kelly case end-to-end and print a one-line summary.

    Single implementation shared by any caller that needs to reproduce a
    case (this module's own demo run, a notebook, a chart script) rather
    than each one re-deriving the same simulate -> price -> solve pipeline.

    Returns:
        The solved KellyResult, for callers that need it (e.g. to chart).
    """
    sim_prices = simulate_terminal_prices(
        spot, volatility, time_to_expiry, risk_free_rate, n_simulations, random_seed=42,
    )
    sim_returns = spread_return(sim_prices, short_strike, long_strike, credit_per_share)
    result = solve_kelly_fraction(sim_returns)
    print(f"[{label}] f* = {result.f_star:.3f}  (paper: {expected_f_star})")
    return result


def main() -> None:
    # Reproduces the paper's Section VII worked example: GS $990/$1000 put
    # spread, 2026-07-20 chain. Strike/credit figures here are PER-SHARE
    # (see the UNITS note on spread_return) -- e.g. credit=2.90, not the
    # paper's total-contract $290. Width is derived inside spread_return
    # from short_strike - long_strike, not passed separately.
    common_args = dict(
        spot=1055.03,
        time_to_expiry=32 / 365,
        risk_free_rate=0.037,
        short_strike=1000.0,
        long_strike=990.0,
        credit_per_share=2.90,   # $19.875 - $16.975 mid-to-mid
        n_simulations=100_000,
    )

    # Case 1: flat average implied vol (paper's first pass, before noting skew)
    run_kelly_case(volatility=0.350, label="Implied-vol case", expected_f_star="~0.000", **common_args)

    # Case 2: subjective realized vol -- the actual defensible-edge case
    subjective_result = run_kelly_case(
        volatility=0.313, label="Subjective-vol case", expected_f_star="~0.075", **common_args,
    )
    print(f"  Half-Kelly = {subjective_result.half_kelly:.3f}   Quarter-Kelly = {subjective_result.quarter_kelly:.3f}")


if __name__ == "__main__":
    main()
