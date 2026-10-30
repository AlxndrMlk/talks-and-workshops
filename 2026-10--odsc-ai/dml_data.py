"""Synthetic data for notebook 03 (DML), "Business Loves Uplift" workshop.

One row per sales rep. The data follows this graph and nothing else:

    experience -> recommender, experience -> sales_post     confounder
    U          -> recommender, U          -> sales_post     hidden confounder
    store_size -> sales_post, footfall -> sales_post        drive sales only
    recommender -> rec_skus -> sales_post                   mediator (the only causal path)
    recommender -> bonus <- sales_post                      collider

Columns:
    recommender     1 if the rep got the recommender (treatment)
    sales_post      mean weekly sales per store after it went live, k$ (outcome)
    experience      years in the job
    store_size      the rep's average store size, sqm
    footfall        the rep's average store footfall, visits/week
    rec_skus        recommended products acted on per store per month
    bonus           1 if the rep has the recommender and reached the sales target
    relationship_strength   hidden confounder (SD units), reps_full only

Separate table for the placebo test in notebook 05:
    sales_placebo   mean weekly sales per store before the recommender went live, k$
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    seed: int = 57
    n_reps: int = 240
    # reps
    exp_mean: float = 6.5          # years of experience
    exp_sd: float = 2.5
    size_sd: float = 0.15
    footfall_sd: float = 0.15
    # who gets the recommender (logit)
    t_intercept: float = -1.0
    t_exp: float = 0.8             # per SD of experience
    t_u: float = 0.8               # per SD of relationship strength
    # sales without the recommender
    base_sales: float = 20.0
    y_exp: float = 0.04            # share of sales per SD of experience
    y_u: float = 0.04              # share of sales per SD of relationship strength
    noise_sd: float = 0.04
    # effect: share of sales, the same CATE as in uplift_data.py
    effect_mean: float = 0.12      # at mean experience
    effect_exp_slope: float = 0.4  # smaller for experienced reps
    effect_floor: float = 0.2
    # mediator: products acted on per store per month
    skus_own: float = 4.0          # on the rep's own judgement, with or without the recommender
    skus_boost: float = 10.0       # extra ones with the recommender
    # collider
    bonus_target: float = 23.0     # k$ per store-week


def cate_pct(experience, p: Params = Params()):
    """True CATE as a share of sales, as a function of rep experience (years)."""
    z = (np.asarray(experience, dtype=float) - p.exp_mean) / p.exp_sd
    return p.effect_mean * np.maximum(p.effect_floor, 1.0 - p.effect_exp_slope * z)


def make_datasets(p: Params = Params()):
    """The observed table, the table with the hidden confounder, and the answer key."""
    rng = np.random.default_rng(p.seed)
    n = p.n_reps

    experience = np.clip(rng.normal(p.exp_mean, p.exp_sd, n), 0.5, 30.0).round(1)
    z = (experience - p.exp_mean) / p.exp_sd
    u = rng.normal(size=n)
    store_size = np.exp(np.log(400) + rng.normal(0, p.size_sd, n)).round(1)
    footfall = np.exp(np.log(1500) + rng.normal(0, p.footfall_sd, n)).round(0)

    recommender = rng.binomial(1, 1 / (1 + np.exp(-(p.t_intercept + p.t_exp * z + p.t_u * u))))

    sales0 = (p.base_sales * (store_size / 400) ** 0.5 * (footfall / 1500) ** 0.4
              * (1 + p.y_exp * z + p.y_u * u) * (1 + rng.normal(0, p.noise_sd, n)))
    rec_skus = rng.poisson(p.skus_own, n) + recommender * rng.poisson(p.skus_boost, n)
    pct = cate_pct(experience, p)
    # the recommender works only through rec_skus
    sales_post = sales0 * (1 + pct * (rec_skus - p.skus_own) / p.skus_boost)
    bonus = ((recommender == 1) & (sales_post > p.bonus_target)).astype(int)

    full = pd.DataFrame({
        "rep_id": np.arange(n),
        "recommender": recommender,
        "sales_post": sales_post.round(3),
        "experience": experience,
        "store_size": store_size,
        "footfall": footfall,
        "rec_skus": rec_skus,
        "bonus": bonus,
        "relationship_strength": u.round(3),
    })
    # before the rollout: the same reps, nobody has the recommender yet
    sales_placebo = (p.base_sales * (store_size / 400) ** 0.5 * (footfall / 1500) ** 0.4
                     * (1 + p.y_exp * z + p.y_u * u) * (1 + rng.normal(0, p.noise_sd, n)))
    placebo = full[["rep_id", "recommender", "experience", "store_size", "footfall"]].assign(
        sales_placebo=sales_placebo.round(3))
    tau = sales0 * pct                 # expected effect per rep, k$
    slope, intercept = np.polyfit(experience, tau, 1)
    truth = {
        "ate_abs": tau.mean(),
        "ate_pct": tau.sum() / sales0.sum(),
        "cate_abs_intercept": intercept,
        "cate_abs_slope_per_year": slope,
    }
    return {
        "reps_observed": full.drop(columns="relationship_strength"),
        "reps_full": full,
        "placebo_observed": placebo,
        "rep_tau": pd.DataFrame({"experience": experience, "true_effect_abs": tau}),
        "truth": truth,
    }


def fingerprint(d):
    """Short checksum to confirm everyone has the same data."""
    return f"{d['reps_observed'].sales_post.sum():.3f}"
