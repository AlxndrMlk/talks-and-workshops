"""Synthetic data for the "Business Loves Uplift" workshop (ODSC AI West 2026).

One world, one timeline:

    8 regions x 30 sales reps x 120-200 stores, weekly sales.

    weeks  0..7    history before the A/B test (no treatment anywhere)
    weeks  8..15   A/B test of the recommender on 40 randomly chosen reps
    weeks 16..23   washout (we assume no carryover: the effect stops when the test stops)
    weeks 24..75   pre-period of the rollout (52 weeks)
    weeks 76..95   post-period: the recommender is live for every store of every
                   North rep (20 weeks)

The rollout data is used twice: synthetic control on region-week sales and DML on
one row per rep. The A/B window is used for the (clustered) A/B test notebooks.

Hidden confounder: rep-store relationship strength. Management piloted in the North
because "the North team is our strongest". Strong relationships make sales grow faster,
and strength is correlated with rep experience. The observed tables drop it.

Interference: a rep who has the recommender on some of their stores applies what they
learn to their other stores too (within-rep spillover, `gamma_within`), and reps in the
same region talk (between-rep spillover, `gamma_between`).

Tables returned by `make_datasets()` (sales in k$ per store per week):

    reps_observed / reps_full   one row per rep, for DML
        north           1 if the rep is in the North (treatment)
        experience      years in the job
        n_stores, store_size (sqm), footfall (visits/week), urban_share
        sales_pre       mean weekly sales per store, weeks 24..75
        sales_post      mean weekly sales per store, weeks 76..95 (outcome)
        new_skus        new SKUs listed per store per month (mediator: bad control)
        bonus           1 if the rep got a management bonus (collider: bad control)
        relationship_strength   hidden confounder, reps_full only
    placebo_observed / placebo_full   one row per rep, for the DML placebo test
        same covariates; sales_pre = weeks 24..67, outcome sales_placebo = weeks 68..75
    region_week     region, week (0 = start of the pre-period, treated from week 52),
                    sales (mean per store), treated
    ab_rep / ab_store   one row per test store, for the A/B notebooks
        treated, rep_treated_share, store covariates,
        sales_pre (weeks 0..7), sales_test (weeks 8..15)
    history         store-week sales, weeks 0..7, all stores (power analysis, A/A test)
    rep_truth       true effect per rep if treated (k$ and %)
    truth           the answer key (see `truth()`)

Every draw comes from `seed`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

REGIONS = ("North", "South", "East", "West", "Central", "Coast", "Lakes", "Highlands")
# Second world for notebook 04: the North grows faster than any donor region, classic SC is
# biased (about +23%) and the fake-date check fires; SC with a linear trend gets close.
# Use it for SC only: the hull miss biases DML too.
HULL_SEED = 51


@dataclass(frozen=True)
class Params:
    seed: int = 23                  # typical world, North inside the donors hull (see the tests)
    # structure
    reps_per_region: int = 30
    stores_min: int = 120
    stores_max: int = 200
    # timeline (weeks)
    n_ab_pre: int = 8
    n_ab_test: int = 8
    n_washout: int = 8
    n_pre: int = 52
    n_post: int = 20
    n_placebo: int = 8             # last pre-period weeks used as the placebo outcome
    n_ab_reps: int = 40
    # reps
    exp_mean: float = 6.5          # years of experience
    exp_sd: float = 2.5
    # North reps are more experienced; so are Coast reps (Coast was the other strong candidate)
    exp_region_shift: tuple = (1.5, 0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0)
    rep_sd: float = 0.04           # rep random effect, share of sales
    rep_week_sd: float = 0.02      # rep-level weekly shocks (routes, absences), AR(1)
    exp_level: float = 0.03        # sales per SD of experience
    exp_growth: float = 0.03       # growth per year per SD of experience (share of average sales)
    rep_growth_sd: float = 0.03    # rep-specific growth per year, unexplained
    # hidden confounder: relationship strength (SD units)
    u_on_exp: float = 0.5
    u_region_shift: tuple = (0.6, -0.3, 0.0, -0.2, 1.1, 1.2, -0.1, -0.4)
    u_level: float = 0.02          # sales per SD of strength
    u_growth: float = 0.05         # growth per year per SD of strength (share of average sales)
    # stores
    base_sales: float = 20.0
    urban_share: tuple = (0.5, 0.65, 0.35, 0.5, 0.3, 0.7, 0.4, 0.55)
    urban_rep_sd: float = 0.2      # reps cover more or less urban territories
    size_sd: float = 0.25
    footfall_sd: float = 0.2
    store_sd: float = 0.08
    noise_sd: float = 0.12         # store-week noise, share of sales
    # time
    season_amp: float = 0.08
    n_factors: int = 3
    factor_sd: float = 0.012
    factor_phi: float = 0.9
    region_noise_sd: float = 0.004
    # mediator: new SKUs a store lists per month after the rollout date. Every new SKU adds
    # effect_mean / skus_boost to sales, so the recommender works only through new listings
    skus_base: float = 6.0
    skus_per_sd_u: float = 0.0
    skus_rep_sd: float = 0.5
    skus_boost: float = 4.0        # extra SKUs a store lists with the recommender
    # collider: management bonus
    bonus_north: float = 0.3
    bonus_noise: float = 0.8
    bonus_cut: float = 1.3
    # effect
    effect_mean: float = 0.12      # average uplift (share of sales) at mean experience
    effect_exp_slope: float = 0.4  # smaller for experienced reps
    effect_floor: float = 0.2
    gamma_within: float = 0.5      # untreated store, fully treated rep -> half the effect
    gamma_between: float = 0.1     # untreated rep, fully treated region -> a tenth

    def __post_init__(self):
        R = len(REGIONS)
        checks = [
            (len(self.u_region_shift) == R and len(self.urban_share) == R and len(self.exp_region_shift) == R,
             f"region tuples need {R} values"),
            (self.reps_per_region >= 2, "need at least 2 reps per region"),
            (1 <= self.stores_min <= self.stores_max, "need 1 <= stores_min <= stores_max"),
            (0 < self.n_placebo < self.n_pre, "need 0 < n_placebo < n_pre"),
            (self.n_post > 0 and self.n_ab_test > 0, "need post-period and A/B test weeks"),
            (2 <= self.n_ab_reps <= R * self.reps_per_region, "n_ab_reps out of range"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

    @property
    def t_ab_start(self):
        return self.n_ab_pre

    @property
    def t_ab_end(self):
        return self.n_ab_pre + self.n_ab_test

    @property
    def t_pre_start(self):
        return self.n_ab_pre + self.n_ab_test + self.n_washout

    @property
    def t_rollout(self):
        return self.t_pre_start + self.n_pre

    @property
    def n_weeks(self):
        return self.t_rollout + self.n_post


def effect_multiplier(experience, p: Params):
    """Relative size of the uplift for a rep with this experience (1 at the mean)."""
    z = (np.asarray(experience, dtype=float) - p.exp_mean) / p.exp_sd
    return np.maximum(p.effect_floor, 1.0 - p.effect_exp_slope * z)


def cate_pct(experience, p: Params = Params()):
    """True CATE as a share of sales, as a function of rep experience (years)."""
    return p.effect_mean * effect_multiplier(experience, p)


def _mix_weights(target, points):
    """Weights >= 0 summing to 1, as close to equal as possible, with weights @ points = target.

    `points`: one row per donor, one column per feature. Closed form with an active set
    (no optimizer, so no dependence on the SciPy version). If the target is outside the
    donors' hull it is matched only approximately (check the miss); far outside, all the
    weight can land on one donor."""
    points = np.asarray(points, dtype=float).reshape(len(points), -1)
    target = np.atleast_1d(np.asarray(target, dtype=float))
    scale = points.std(0) + 1e-12
    pts, tgt = points / scale, target / scale
    k = len(pts)
    active = np.ones(k, dtype=bool)
    v = np.ones(1)
    while active.sum() > 1:
        n = active.sum()
        C = np.vstack([np.ones(n), pts[active].T])        # sum to 1, match each feature
        b = np.r_[1.0, tgt]
        u = np.full(n, 1 / n)
        v = u + np.linalg.pinv(C) @ (b - C @ u)            # smallest move away from equal weights
        if np.all(v >= -1e-12):
            break
        active[np.flatnonzero(active)[np.argmin(v)]] = False   # drop one donor at a time
    if active.sum() == 1:
        v = np.ones(1)
    w = np.zeros(k)
    w[active] = np.clip(v, 0, None)
    w = np.round(w / w.sum(), 6)
    w = w / w.sum()
    if not np.all(np.isfinite(w)):
        raise FloatingPointError("mix weights are not finite")
    return w


class World:
    """Units, their traits and the no-treatment sales path. Treatment comes later."""

    def __init__(self, p: Params = Params()):
        self.p = p
        rng = np.random.default_rng(p.seed)
        R, n_rep = len(REGIONS), p.reps_per_region * len(REGIONS)

        # reps
        self.rep_region = np.repeat(np.arange(R), p.reps_per_region)
        experience = rng.normal(p.exp_mean + np.asarray(p.exp_region_shift)[self.rep_region], p.exp_sd)
        self.rep_experience = np.clip(experience, 0.5, 30.0).round(1)
        exp_z = (self.rep_experience - p.exp_mean) / p.exp_sd
        self.rep_strength = (p.u_on_exp * exp_z + np.asarray(p.u_region_shift)[self.rep_region]
                             + np.sqrt(1 - p.u_on_exp ** 2) * rng.normal(size=n_rep))
        rep_re = rng.normal(0, p.rep_sd, n_rep)
        rep_mult = 1 + p.exp_level * exp_z + p.u_level * self.rep_strength + rep_re
        rep_growth = (p.u_growth * self.rep_strength + p.exp_growth * exp_z
                      + rng.normal(0, p.rep_growth_sd, n_rep))           # per year, share of avg sales

        # stores, contiguous by rep (and so by region)
        self.rep_n_stores = rng.integers(p.stores_min, p.stores_max + 1, n_rep)
        self.store_rep = np.repeat(np.arange(n_rep), self.rep_n_stores)
        self.store_region = self.rep_region[self.store_rep]
        S = len(self.store_rep)
        rep_urban = np.clip(np.asarray(p.urban_share)[self.rep_region] + rng.normal(0, p.urban_rep_sd, n_rep),
                            0.05, 0.95)
        self.store_urban = rng.random(S) < rep_urban[self.store_rep]
        self.store_size = np.exp(np.log(400) + 0.15 * self.store_urban + rng.normal(0, p.size_sd, S)).round()
        self.store_footfall = np.exp(np.log(1500) + 0.4 * self.store_urban + 0.3 * np.log(self.store_size / 400)
                                     + rng.normal(0, p.footfall_sd, S)).round()
        u_s = self.rep_strength[self.store_rep]
        skus_rep = rng.normal(0, p.skus_rep_sd, n_rep)
        self.skus0 = np.maximum(0, p.skus_base + p.skus_per_sd_u * u_s + skus_rep[self.store_rep]
                                + rng.normal(0, 1.5, S))
        mult = effect_multiplier(self.rep_experience[self.store_rep], p)
        self.skus_gain = p.skus_boost * mult * rng.gamma(20, 1 / 20, S)
        sku_effect = p.effect_mean / p.skus_boost                     # sales share per listed SKU
        self.base = (p.base_sales * (self.store_size / 400) ** 0.5 * (self.store_footfall / 1500) ** 0.4
                     * np.exp(rng.normal(0, p.store_sd, S)) * rep_mult[self.store_rep])

        # uplift if treated, as a share of sales; it works through the new SKUs
        self.store_pct = sku_effect * self.skus_gain

        # time: seasonality, region factors, growth
        T = p.n_weeks
        t = np.arange(T)
        season = p.season_amp * np.sin(2 * np.pi * t / 52)
        F = np.zeros((p.n_factors, T))
        for k in range(1, T):
            F[:, k] = p.factor_phi * F[:, k - 1] + rng.normal(0, 1, p.n_factors)
        F *= p.factor_sd * np.sqrt(1 - p.factor_phi ** 2)
        load = rng.uniform(0.0, 2.0, (R, p.n_factors))
        # The North's factor loadings are a mix of the other regions' loadings: the mix closest to
        # equal weights that also reproduces the North's sales level and growth. With the
        # adjustment below, the North still moves with the rest of the country, so DML sees no
        # North-only shock. The North's level and growth are NOT forced inside the donors' hull:
        # in many seeds no mix of donors matches both (north_mix_miss > 0), and classic synthetic
        # control is biased there. Notebook 04 shows this and motivates another estimator.
        g_scale = self.base.mean()                                    # growth in k$, same for every store
        n_reg = np.bincount(self.store_region)
        level = np.bincount(self.store_region, weights=self.base) / n_reg
        slope = g_scale * np.bincount(self.store_region, weights=rep_growth[self.store_rep]) / n_reg
        pts = np.c_[level, slope]
        self.north_mix = _mix_weights(pts[0], pts[1:])
        # how far the best mix misses the North: level in k$/week, growth in k$/week per year
        self.north_mix_miss_level, self.north_mix_miss_growth = np.abs(self.north_mix @ pts[1:] - pts[0])
        self.north_mix_miss = max(self.north_mix_miss_level, self.north_mix_miss_growth)
        # adjust the donors' loadings so that this mix and equal weights give the same path
        # (adjusted loadings can be negative; that's fine)
        dev = self.north_mix - 1 / (R - 1)
        if dev @ dev > 0:
            load[1:] -= np.outer(dev, dev @ load[1:]) / (dev @ dev)
        load[0] = self.north_mix @ load[1:]
        reg_noise = np.zeros((R, T))
        for k in range(1, T):
            reg_noise[:, k] = 0.5 * reg_noise[:, k - 1] + rng.normal(0, p.region_noise_sd, R)
        self.region_path = load @ F + reg_noise                       # R x T
        rep_wk = np.zeros((n_rep, T))
        for k in range(1, T):
            rep_wk[:, k] = 0.7 * rep_wk[:, k - 1] + rng.normal(0, p.rep_week_sd * np.sqrt(1 - 0.49), n_rep)
        # growth is centred on the middle of the pre-period, so the pre-period average
        # says little about how fast a rep's stores grow
        years = (t - (p.t_pre_start + p.t_rollout) / 2) / 52
        # new SKUs that stores list anyway after the rollout date raise their sales from then on
        listed = np.where(t >= p.t_rollout, 1.0, 0.0)
        self.mu0 = (self.base[:, None]
                    * (1 + season[None, :] + self.region_path[self.store_region] + rep_wk[self.store_rep]
                       + sku_effect * (self.skus0 - p.skus_base)[:, None] * listed[None, :])
                    + g_scale * rep_growth[self.store_rep][:, None] * years[None, :])
        self.eps = rng.normal(0, p.noise_sd, (S, T))
        self.rep_growth = rep_growth
        self.t = t

    @property
    def n_stores(self):
        return len(self.store_rep)


def simulate(world: World, treated: np.ndarray):
    """Sales under a treatment assignment `treated` (stores x weeks, bool).

    Returns (sales, tau): observed sales and the expected effect per store-week
    (including spillovers onto untreated stores). No carryover: the effect in a week
    depends only on who is treated that week.
    """
    p = world.p
    assert treated.shape == world.mu0.shape
    tr = treated.astype(float)
    # share of each rep's stores treated, each week
    rep_share = np.add.reduceat(tr, np.r_[0, np.cumsum(world.rep_n_stores)[:-1]], axis=0) \
        / world.rep_n_stores[:, None]
    # share of the OTHER reps in the region with any treated store
    rep_on = (rep_share > 0).astype(float)
    reg_on = np.zeros((len(REGIONS), tr.shape[1]))
    np.add.at(reg_on, world.rep_region, rep_on)
    reg_cnt = np.bincount(world.rep_region)[:, None]
    other_share = (reg_on[world.rep_region] - rep_on) / (reg_cnt[world.rep_region] - 1)
    spill = np.minimum(1.0, p.gamma_within * rep_share + p.gamma_between * other_share)[world.store_rep]
    eff = world.store_pct[:, None] * (tr + (1 - tr) * spill)
    tau = world.mu0 * eff
    sales = (world.mu0 + tau) * (1 + world.eps)
    return sales, tau


def rollout_assignment(world: World):
    tr = np.zeros((world.n_stores, world.p.n_weeks), dtype=bool)
    tr[world.store_region == 0, world.p.t_rollout:] = True
    return tr


def ab_assignment(world: World, unit="rep", n_reps=None, seed=None):
    """A/B test in the history window. `unit="rep"`: randomize reps (all their stores
    get the same arm). `unit="store"`: same test reps, randomize their stores."""
    p = world.p
    n_reps = p.n_ab_reps if n_reps is None else n_reps
    rng = np.random.default_rng(p.seed + 1 if seed is None else seed)
    reps = rng.choice(len(world.rep_region), n_reps, replace=False)
    in_test = np.isin(world.store_rep, reps)
    if unit == "rep":
        arm_rep = np.zeros(len(world.rep_region), dtype=bool)
        arm_rep[rng.permutation(reps)[: n_reps // 2]] = True
        arm = arm_rep[world.store_rep]
    elif unit == "store":
        arm = in_test & (rng.random(world.n_stores) < 0.5)
    else:
        raise ValueError("unit must be 'rep' or 'store'")
    tr = np.zeros((world.n_stores, p.n_weeks), dtype=bool)
    tr[arm, p.t_ab_start:p.t_ab_end] = True
    return tr, in_test, arm


def _rep_mean(world, x):
    return np.bincount(world.store_rep, weights=x) / world.rep_n_stores


def _rep_covariates(world):
    return pd.DataFrame({
        "rep_id": np.arange(len(world.rep_region)),
        "region": np.asarray(REGIONS)[world.rep_region],
        "north": (world.rep_region == 0).astype(int),
        "experience": world.rep_experience,
        "n_stores": world.rep_n_stores,
        "store_size": _rep_mean(world, world.store_size).round(1),
        "footfall": _rep_mean(world, world.store_footfall).round(0),
        "urban_share": _rep_mean(world, world.store_urban.astype(float)).round(3),
    })


def rep_table(world: World, sales: np.ndarray, hide_confounder=True):
    """One row per rep for DML. Outcome: mean weekly sales per store after the rollout."""
    p = world.p
    pre, post = slice(p.t_pre_start, p.t_rollout), slice(p.t_rollout, p.n_weeks)
    skus = world.skus0 + np.where(world.store_region == 0, world.skus_gain, 0.0)
    df = _rep_covariates(world)
    df["sales_pre"] = _rep_mean(world, sales[:, pre].mean(1)).round(3)
    df["sales_post"] = _rep_mean(world, sales[:, post].mean(1)).round(3)
    df["new_skus"] = _rep_mean(world, skus).round(2)
    # collider: management bonus for pilot reps and for reps whose sales grew
    rng = np.random.default_rng(p.seed + 7)
    growth = df.sales_post / df.sales_pre - 1
    gz = (growth - growth.mean()) / growth.std()
    df["bonus"] = (p.bonus_north * df.north + gz + rng.normal(0, p.bonus_noise, len(df)) > p.bonus_cut).astype(int)
    if not hide_confounder:
        df["relationship_strength"] = world.rep_strength.round(3)
    return df


def placebo_table(world: World, sales: np.ndarray, hide_confounder=True):
    """One row per rep for the DML placebo test: the 'outcome' is the last pre-period
    weeks, before anyone had the recommender, so the true effect is zero."""
    p = world.p
    early = slice(p.t_pre_start, p.t_rollout - p.n_placebo)
    late = slice(p.t_rollout - p.n_placebo, p.t_rollout)
    df = _rep_covariates(world)
    df["sales_pre"] = _rep_mean(world, sales[:, early].mean(1)).round(3)
    df["sales_placebo"] = _rep_mean(world, sales[:, late].mean(1)).round(3)
    if not hide_confounder:
        df["relationship_strength"] = world.rep_strength.round(3)
    return df


def region_week(world: World, sales: np.ndarray):
    """Mean weekly sales per store, by region, over the rollout window (for SC)."""
    p = world.p
    starts = np.r_[0, np.flatnonzero(np.diff(world.store_region)) + 1]
    m = np.add.reduceat(sales, starts, axis=0) / np.bincount(world.store_region)[:, None]
    m = m[:, p.t_pre_start:]
    R, W = m.shape
    return pd.DataFrame({
        "region": np.repeat(np.asarray(REGIONS), W),
        "week": np.tile(np.arange(W), R),
        "sales": m.ravel().round(4),
        "treated": ((np.repeat(np.arange(R), W) == 0) & (np.tile(np.arange(W), R) >= p.n_pre)).astype(int),
    })


def ab_table(world: World, sales: np.ndarray, in_test, arm):
    """One row per test store: mean weekly sales before and during the test."""
    p = world.p
    idx = np.flatnonzero(in_test)
    rep = world.store_rep[idx]
    share = np.bincount(world.store_rep, weights=arm.astype(float)) / world.rep_n_stores
    return pd.DataFrame({
        "store_id": idx,
        "rep_id": rep,
        "region": np.asarray(REGIONS)[world.store_region[idx]],
        "treated": arm[idx].astype(int),
        "rep_treated_share": share[rep].round(3),
        "experience": world.rep_experience[rep],
        "store_size": world.store_size[idx],
        "footfall": world.store_footfall[idx],
        "urban": world.store_urban[idx].astype(int),
        "sales_pre": sales[idx, :p.t_ab_start].mean(1).round(3),
        "sales_test": sales[idx, p.t_ab_start:p.t_ab_end].mean(1).round(3),
    })


def history(world: World, sales: np.ndarray):
    """Store-week sales of all stores before the A/B test (no treatment anywhere)."""
    p = world.p
    S, W = world.n_stores, p.t_ab_start
    return pd.DataFrame({
        "store_id": np.repeat(np.arange(S), W),
        "rep_id": np.repeat(world.store_rep, W),
        "region": np.asarray(REGIONS)[np.repeat(world.store_region, W)],
        "week": np.tile(np.arange(W), S),
        "sales": sales[:, :W].ravel().round(3),
    })


def rep_truth(world: World):
    """True effect per rep over the post-period if all their stores were treated."""
    p = world.p
    post = slice(p.t_rollout, p.n_weeks)
    mu0 = world.mu0[:, post].mean(1)
    eff = _rep_mean(world, world.store_pct * mu0)
    return pd.DataFrame({
        "rep_id": np.arange(len(world.rep_region)),
        "north": (world.rep_region == 0).astype(int),
        "experience": world.rep_experience,
        "true_effect_abs": eff.round(4),
        "true_effect_pct": (eff / _rep_mean(world, mu0)).round(5),
    })


def truth(world: World, tau, ab=None):
    """Answer key. `abs` in k$ per store-week, `pct` relative to sales without treatment.

    `tau`: effects from the main timeline (rollout). `ab`: {"rep": (tau, in_test, arm), ...},
    one entry per A/B design, each with the effects of its own simulation.
    The CATE entries are the best linear fit in experience over reps: approximately what
    LinearDML on sales levels with X=experience targets (the true CATE has a floor)."""
    p = world.p
    post = slice(p.t_rollout, p.n_weeks)
    ns = world.store_region == 0
    nr = world.rep_region == 0
    mu0_post = world.mu0[:, post].mean(1)
    tau_post = tau[:, post].mean(1)
    rep_tau, rep_mu0 = _rep_mean(world, tau_post), _rep_mean(world, mu0_post)
    rt = rep_truth(world)
    slope, intercept = np.polyfit(rt.experience, rt.true_effect_abs, 1)
    out = {
        # SC target: North region, store-weighted
        "att_north_store_abs": tau_post[ns].mean(),
        "att_north_store_pct": tau_post[ns].sum() / mu0_post[ns].sum(),
        # DML targets: one row per rep
        "att_north_rep_abs": rep_tau[nr].mean(),
        "att_north_rep_pct": rep_tau[nr].mean() / rep_mu0[nr].mean(),
        "ate_rep_abs": rt.true_effect_abs.mean(),
        "ate_rep_pct": rt.true_effect_abs.mean() / rep_mu0.mean(),
        "atc_rep_abs": rt.true_effect_abs[~nr].mean(),
        # best linear CATE in experience, k$ (what LinearDML on levels with X=experience targets)
        "cate_abs_intercept": intercept,
        "cate_abs_slope_per_year": slope,
        "rollout_spillover_outside_north": float(np.abs(tau[~ns][:, post]).max()),
    }
    win = slice(p.t_ab_start, p.t_ab_end)
    mu_ab = world.mu0[:, win].mean(1)
    for name, (tau_ab, in_test, arm) in (ab or {}).items():
        t_ab = tau_ab[:, win].mean(1)
        trt, ctl = in_test & arm, in_test & ~arm
        diff = t_ab[trt].mean() - t_ab[ctl].mean()
        out.update({
            f"ab_{name}_direct_abs": t_ab[trt].mean(),           # effect on treated stores
            f"ab_{name}_control_spill_abs": t_ab[ctl].mean(),    # leaked onto controls
            f"ab_{name}_expected_diff_abs": diff,                # what treated - control estimates
            f"ab_{name}_direct_pct": t_ab[trt].sum() / mu_ab[trt].sum(),
            f"ab_{name}_expected_diff_pct": diff / (mu_ab[ctl].mean() + t_ab[ctl].mean()),
        })
    return {k: float(v) for k, v in out.items()}


def make_datasets(p: Params = Params()):
    """Everything the notebooks need, as DataFrames plus the answer key."""
    w = World(p)
    roll = rollout_assignment(w)
    tr_rep, in_rep, arm_rep = ab_assignment(w, "rep")
    tr_sto, in_sto, arm_sto = ab_assignment(w, "store")
    sales, tau = simulate(w, roll | tr_rep)           # the main timeline
    sales_sto, tau_sto = simulate(w, roll | tr_sto)   # the same world with a store-level A/B test
    t = truth(w, tau, {"rep": (tau, in_rep, arm_rep), "store": (tau_sto, in_sto, arm_sto)})
    return {
        "reps_observed": rep_table(w, sales, hide_confounder=True),
        "reps_full": rep_table(w, sales, hide_confounder=False),
        "placebo_observed": placebo_table(w, sales, hide_confounder=True),
        "placebo_full": placebo_table(w, sales, hide_confounder=False),
        "region_week": region_week(w, sales),
        "ab_rep": ab_table(w, sales, in_rep, arm_rep),
        "ab_store": ab_table(w, sales_sto, in_sto, arm_sto),
        "history": history(w, sales),
        "rep_truth": rep_truth(w),
        "truth": t,
        "world": w,
    }


def fingerprint(d):
    """Short checksum to confirm everyone has the same data (prints the same in Colab)."""
    return f"{d['region_week'].sales.sum():.3f}/{d['reps_observed'].sales_post.sum():.3f}"


TABLES = ("reps_observed", "reps_full", "placebo_observed", "placebo_full", "region_week",
          "ab_rep", "ab_store", "history", "rep_truth")


def save_datasets(outdir="data", p: Params = Params()):
    os.makedirs(outdir, exist_ok=True)
    d = make_datasets(p)
    for k in TABLES:
        d[k].to_csv(os.path.join(outdir, f"{k}.csv"), index=False)
    with open(os.path.join(outdir, "truth.json"), "w") as f:
        json.dump({"params": asdict(p), "fingerprint": fingerprint(d), "truth": d["truth"]}, f, indent=2)
    return d


if __name__ == "__main__":
    d = make_datasets()
    print(fingerprint(d))
    print(json.dumps(d["truth"], indent=2))
