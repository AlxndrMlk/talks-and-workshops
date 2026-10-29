"""Tests for uplift_data.py. Run: python -m pytest -q test_uplift_data.py

Lessons are checked on the default seed AND on average over seeds 2026-2035, so the
default seed is not a lucky draw.
"""
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy.optimize import minimize

import uplift_data as ud

X = "experience + store_size + footfall + urban_share + sales_pre"
SEEDS = range(2026, 2036)


@pytest.fixture(scope="module")
def d():
    return ud.make_datasets()


def north_shock(w):
    """How much more the North's regional path moves pre -> post than the donors' (k$)."""
    p, rp = w.p, w.region_path
    pre, post = slice(p.t_pre_start, p.t_rollout), slice(p.t_rollout, p.n_weeks)
    move = rp[:, post].mean(1) - rp[:, pre].mean(1)
    return (move[0] - move[1:].mean()) * w.base.mean()


@pytest.fixture(scope="module")
def many():
    """Ten worlds, without the big arrays (keeps memory low)."""
    out = []
    for s in SEEDS:
        m = ud.make_datasets(ud.Params(seed=s))
        w = m.pop("world")
        m.update(p=w.p, miss=w.north_mix_miss, shock=north_shock(w), mix=w.north_mix)
        out.append(m)
    return out


def P(d):
    return d["p"] if "p" in d else d["world"].p


def icc(df, y="sales_test"):
    g = df.groupby("rep_id")[y]
    k = g.size().mean()
    msb, msw = g.mean().var() * k, g.var().mean()
    return (msb - msw) / (msb + (k - 1) * msw)


def north_coef(df, formula):
    return smf.ols(formula, df).fit().params["north"]


def dml_obs(d):
    return north_coef(d["reps_observed"], "sales_post ~ north + " + X)


def dml_full(d):
    return north_coef(d["reps_full"], "sales_post ~ north + " + X + " + relationship_strength")


def synth(region_week, n_pre, intercept=True):
    """Synthetic control: weights >= 0 summing to 1, optionally plus an intercept (demeaned)."""
    m = region_week.pivot(index="week", columns="region", values="sales")
    y, D = m["North"].to_numpy(), m.drop(columns="North").to_numpy()
    if intercept:
        y, D = y - y[:n_pre].mean(), D - D[:n_pre].mean(0)
    k = D.shape[1]
    res = minimize(lambda v: ((y[:n_pre] - D[:n_pre] @ v) ** 2).sum(), np.ones(k) / k,
                   bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
                   method="SLSQP")
    return y - D @ res.x


def synth_trend(region_week, n_pre):
    """Synthetic control on series detrended over the pre-period (level + linear trend)."""
    m = region_week.pivot(index="week", columns="region", values="sales")
    Z = np.c_[np.ones(len(m)), np.arange(len(m))]
    detrend = lambda v: v - Z @ np.linalg.lstsq(Z[:n_pre], v[:n_pre], rcond=None)[0]
    y = detrend(m["North"].to_numpy())
    D = np.column_stack([detrend(c) for c in m.drop(columns="North").to_numpy().T])
    k = D.shape[1]
    res = minimize(lambda v: ((y[:n_pre] - D[:n_pre] @ v) ** 2).sum(), np.ones(k) / k,
                   bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda v: v.sum() - 1},
                   method="SLSQP")
    return y - D @ res.x


def sc_att(d, intercept=True):
    p = P(d)
    return synth(d["region_week"], p.n_pre, intercept)[p.n_pre:].mean()


def fake_date_gap(d, weeks_before=20):
    p = P(d)
    rw = d["region_week"][d["region_week"].week < p.n_pre]
    return synth(rw, p.n_pre - weeks_before)[p.n_pre - weeks_before:].mean()


# --- structure and API -----------------------------------------------------------

def test_structure(d):
    w, p = d["world"], d["world"].p
    assert np.all(np.bincount(w.rep_region) == p.reps_per_region)
    assert w.rep_n_stores.min() >= p.stores_min and w.rep_n_stores.max() <= p.stores_max
    assert w.mu0.shape == (w.n_stores, p.n_weeks) and np.all(w.mu0 > 0)
    rw = d["region_week"]
    assert rw.week.nunique() == p.n_pre + p.n_post
    assert rw.loc[rw.treated == 1, "region"].unique().tolist() == ["North"]
    assert rw.loc[rw.treated == 1, "week"].min() == p.n_pre
    assert len(d["history"]) == w.n_stores * p.t_ab_start
    assert len(d["ab_rep"]) == len(d["ab_store"]) == np.isin(w.store_rep, d["ab_rep"].rep_id.unique()).sum()
    assert len(d["rep_truth"]) == len(d["reps_observed"])


def test_params_validation():
    with pytest.raises(ValueError):
        ud.Params(reps_per_region=1)
    with pytest.raises(ValueError):
        ud.Params(urban_share=(0.5,) * 7)
    with pytest.raises(ValueError):
        ud.Params(n_placebo=52)


def test_reproducible():
    a, b = ud.World(), ud.World()
    assert np.array_equal(a.mu0, b.mu0) and np.array_equal(a.eps, b.eps)
    assert not np.allclose(a.base[:100], ud.World(ud.Params(seed=7)).base[:100])


GOLDEN = {"base0": 19.26147443254579, "att_north_store_abs": 2.2597992864992302,
          "fingerprint": "13302.553/5496.520"}   # numpy 2.2.4


def test_golden_values(d):
    """Catches a change in numpy's random streams (e.g. a different numpy in Colab)."""
    assert d["world"].base[0] == pytest.approx(GOLDEN["base0"], rel=1e-6)
    assert d["truth"]["att_north_store_abs"] == pytest.approx(GOLDEN["att_north_store_abs"], rel=1e-6)
    assert ud.fingerprint(d) == GOLDEN["fingerprint"]


def test_observed_tables_hide_only_the_confounder(d):
    for obs, full in (("reps_observed", "reps_full"), ("placebo_observed", "placebo_full")):
        o, f = d[obs], d[full]
        assert "relationship_strength" not in o.columns
        assert set(f.columns) - set(o.columns) == {"relationship_strength"}
        assert o.equals(f[o.columns])
    # the placebo window is not in the DML table (it would let attendees recover the trend)
    assert "sales_placebo" not in d["reps_observed"].columns


def test_save_datasets(tmp_path):
    ud.save_datasets(tmp_path)
    for f in [f"{k}.csv" for k in ud.TABLES] + ["truth.json"]:
        assert (tmp_path / f).exists()
    meta = json.loads((tmp_path / "truth.json").read_text())
    assert {"params", "truth", "fingerprint"} <= meta.keys()


def test_tables_match_groupby(d):
    """The reduceat/bincount shortcuts agree with plain pandas."""
    w, p = d["world"], d["world"].p
    sales, _ = ud.simulate(w, ud.rollout_assignment(w) | ud.ab_assignment(w, "rep")[0])
    store_post = pd.Series(sales[:, p.t_rollout:].mean(1))
    by_rep = store_post.groupby(w.store_rep).mean().round(3).to_numpy()
    assert np.allclose(by_rep, d["reps_observed"].sales_post, atol=1e-3)
    wk = 60
    by_reg = pd.Series(sales[:, p.t_pre_start + wk]).groupby(w.store_region).mean().to_numpy()
    rw = d["region_week"]
    assert np.allclose(by_reg, rw[rw.week == wk].sales.to_numpy(), atol=1e-4)


def test_effect_only_inside_the_windows():
    w = ud.World()
    p = w.p
    _, tau = ud.simulate(w, ud.rollout_assignment(w) | ud.ab_assignment(w, "rep")[0])
    active = np.abs(tau).sum(0) > 0
    expected = np.zeros(p.n_weeks, bool)
    expected[p.t_ab_start:p.t_ab_end] = True
    expected[p.t_rollout:] = True
    assert np.array_equal(active, expected)


def test_ab_design_does_not_change_rollout_data():
    w = ud.World()
    roll = ud.rollout_assignment(w)
    s1, _ = ud.simulate(w, roll | ud.ab_assignment(w, "rep")[0])
    s2, _ = ud.simulate(w, roll | ud.ab_assignment(w, "store")[0])
    assert ud.region_week(w, s1).equals(ud.region_week(w, s2))
    assert ud.rep_table(w, s1).equals(ud.rep_table(w, s2))


# --- clustering ------------------------------------------------------------------

def test_icc_is_realistic(d, many):
    assert 0.08 < icc(d["ab_rep"]) < 0.25
    assert 0.12 < np.mean([icc(m["ab_rep"]) for m in many]) < 0.18


def test_clustering_matters_in_aa(d):
    """A/A on the pre-test history: store-level SEs over-reject, rep-clustered SEs don't (much)."""
    h = d["history"]
    y = h.groupby("store_id").sales.mean()
    rep = h.groupby("store_id").rep_id.first()
    rng = np.random.default_rng(0)
    n, rej_iid, rej_cl = 200, 0, 0
    reps_all = np.unique(rep)
    for _ in range(n):
        reps = rng.choice(reps_all, 40, replace=False)
        sel = rep.isin(reps)
        df = pd.DataFrame({"y": y[sel], "t": rep[sel].isin(reps[:20]).astype(int), "g": rep[sel]})
        fit = smf.ols("y ~ t", df).fit()
        rej_iid += fit.pvalues["t"] < 0.05
        rej_cl += fit.get_robustcov_results("cluster", groups=df.g).pvalues[1] < 0.05
    assert rej_iid / n > 0.3
    assert rej_cl / n < 0.12


# --- hidden confounder -----------------------------------------------------------

def test_confounder_relations(d):
    full = d["reps_full"]
    r = np.corrcoef(full.relationship_strength, full.experience)[0, 1]
    assert 0.3 < r < 0.7
    by = full.groupby("north").relationship_strength.mean()
    assert by[1] - by[0] > 0.4                     # the North team is strong...
    region_u = full.groupby("region").relationship_strength.mean()
    assert region_u.max() > region_u["North"]      # ...but Coast is stronger (SC hull)


def test_hull_violation_biases_sc(many):
    """In some worlds no mix of donors matches the North's level and growth together
    (north_mix_miss > 0). Classic SC is biased there: notebook 04 shows it."""
    for m in many:
        rw = m["region_week"]
        lvl = rw[rw.week < P(m).n_pre].groupby("region").sales.mean()
        assert lvl.drop("North").max() > lvl["North"] > lvl.drop("North").min()
    out = [m for m in many if m["miss"] > 0.05]
    inside = [m for m in many if m["miss"] <= 0.05]
    assert 1 <= len(out) <= 6
    err = lambda ms: np.mean([sc_att(m) / m["truth"]["att_north_store_abs"] - 1 for m in ms])
    assert err(out) > err(inside) + 0.05


def test_north_moves_with_the_country(d, many):
    """No North-only regional shock that DML could mistake for the effect."""
    assert d["world"].north_mix.max() < 0.8          # not a copy of one donor
    assert all(abs(m["shock"]) < 0.15 for m in many)


def test_mix_weights():
    s = np.array([-1.0, 0.0, 1.0, 2.0])
    w = ud._mix_weights(0.5, s)
    assert w.min() >= 0 and w.sum() == pytest.approx(1) and w @ s == pytest.approx(0.5, abs=1e-5)
    assert np.allclose(ud._mix_weights(s.mean(), s), 0.25)
    w = ud._mix_weights(1.9, s)                       # needs mostly the top donors
    assert w @ s == pytest.approx(1.9, abs=1e-5) and w[0] == 0
    assert ud._mix_weights(5.0, s) @ s == pytest.approx(2.0)   # outside: closest possible


def test_mix_weights_never_nan():
    """Targets far outside the hull, any number of donors and features: finite weights."""
    rng = np.random.default_rng(0)
    for _ in range(3000):
        k, f = rng.integers(2, 9), rng.integers(1, 3)
        pts = rng.normal(size=(k, f))
        w = ud._mix_weights(rng.normal(0, 3, f), pts)
        assert np.all(np.isfinite(w)) and w.min() >= 0 and w.sum() == pytest.approx(1)


def test_positivity(d):
    from sklearn.linear_model import LogisticRegression
    obs = d["reps_observed"]
    cols = ["experience", "store_size", "footfall", "urban_share", "sales_pre"]
    Z = (obs[cols] - obs[cols].mean()) / obs[cols].std()
    ps = LogisticRegression(C=1.0).fit(Z, obs.north).predict_proba(Z)[:, 1]
    assert ps[obs.north == 1].min() < 0.5          # some North reps look like the rest
    assert ps[obs.north == 0].max() > 0.2          # some other reps look like the North
    assert ps[obs.north == 1].max() < 0.95         # ATT needs a comparable rep for every North rep


def test_observed_dml_is_biased_up_full_is_close(d):
    tr = d["truth"]["att_north_rep_abs"]
    assert dml_obs(d) > 1.15 * tr                  # hidden confounder pushes the estimate up
    assert abs(dml_full(d) - tr) < 0.1 * tr        # with it, we are close
    assert dml_obs(d) > dml_full(d)


def test_confounder_bias_across_seeds(many):
    obs = np.array([dml_obs(m) / m["truth"]["att_north_rep_abs"] - 1 for m in many])
    full = np.array([dml_full(m) / m["truth"]["att_north_rep_abs"] - 1 for m in many])
    assert 0.15 < obs.mean() < 0.4                 # +15..40% on average
    assert abs(full.mean()) < 0.12                 # close, not exact
    assert np.all(obs > full)


def test_no_treatment_world_has_no_north_gap_given_confounder(d):
    w = d["world"]
    nf = ud.rep_table(w, w.mu0, hide_confounder=False)   # expected sales, nobody treated
    assert abs(north_coef(nf, "sales_post ~ north + " + X + " + relationship_strength")) < 0.2


def _bound(m_y, m_d, k):
    """Cinelli & Hazlett bias bound for a confounder k times as strong as experience."""
    def pr2(m, v):
        t = m.tvalues[v]
        return t ** 2 / (t ** 2 + m.df_resid)
    r2yx, r2dx = pr2(m_y, "experience"), pr2(m_d, "experience")
    r2dz = k * r2dx / (1 - r2dx)
    r2zx = k * r2dx ** 2 / ((1 - k * r2dx) * (1 - r2dx))      # as in sensemakr
    r2yz = min(1, ((np.sqrt(k) + np.sqrt(r2zx)) / np.sqrt(1 - r2zx)) ** 2 * (r2yx / (1 - r2yx)))
    return m_y.bse["north"] * np.sqrt(r2yz * r2dz / (1 - r2dz)) * np.sqrt(m_y.df_resid)


def test_sensitivity_benchmarks_bracket_the_bias(d, many):
    """The hidden confounder is a few times as strong as experience: the 1x benchmark
    understates the bias, a 4x benchmark covers it."""
    def check(m):
        obs = m["reps_observed"]
        m_y = smf.ols("sales_post ~ north + " + X, obs).fit()
        m_d = smf.ols("north ~ " + X, obs).fit()
        bias = dml_obs(m) - dml_full(m)
        return _bound(m_y, m_d, 1) < bias, bias < _bound(m_y, m_d, 4)
    assert all(check(d))
    res = np.array([check(m) for m in many])
    assert res[:, 0].mean() >= 0.7 and res[:, 1].mean() >= 0.7


def test_placebo_flags_the_confounder(d, many):
    f_obs = "sales_placebo ~ north + " + X
    f_full = f_obs + " + relationship_strength"
    po, pf = d["placebo_observed"], d["placebo_full"]
    obs = north_coef(po, f_obs)
    assert obs > 0.2
    assert abs(north_coef(pf, f_full)) < obs
    assert np.mean([north_coef(x["placebo_observed"], f_obs) for x in many]) > 0.2
    assert abs(np.mean([north_coef(x["placebo_full"], f_full) for x in many])) < 0.1


def test_shuffled_treatment_is_null(d):
    obs = d["reps_observed"].copy()
    rng = np.random.default_rng(1)
    est = []
    for _ in range(100):
        obs["north"] = rng.permutation(obs.north.to_numpy())
        est.append(north_coef(obs, "sales_post ~ north + " + X))
    assert abs(np.mean(est)) < 0.1


def test_econml_linear_dml(d):
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t = d["truth"]
    W = ["store_size", "footfall", "urban_share", "sales_pre", "n_stores"]

    def fit(df, extra, y="sales_post"):
        m = LinearDML(model_y=make_pipeline(StandardScaler(), LassoCV()),
                      model_t=make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=2000)),
                      discrete_treatment=True, cv=5, random_state=0)
        m.fit(df[y], df.north, X=df[["experience"]], W=df[W + extra])
        Xn = df.loc[df.north == 1, ["experience"]]
        return m, m.ate(X=Xn), m.ate_interval(X=Xn)

    m_obs, att_obs, _ = fit(d["reps_observed"], [])
    m_full, att_full, _ = fit(d["reps_full"], ["relationship_strength"])
    assert att_obs > 1.15 * t["att_north_rep_abs"]
    assert abs(att_full - t["att_north_rep_abs"]) < 0.1 * t["att_north_rep_abs"]
    assert m_full.coef_[0] == pytest.approx(t["cate_abs_slope_per_year"], abs=0.08)
    # placebo: the observed analysis finds a "pre-period effect", the full one doesn't
    _, plc, (lo, _) = fit(d["placebo_observed"], [], "sales_placebo")
    _, plc_f, _ = fit(d["placebo_full"], ["relationship_strength"], "sales_placebo")
    assert lo > 0 and abs(plc_f) < 0.5 * plc


# --- mediator and collider -------------------------------------------------------

def test_mediator_blocks_the_effect(d, many):
    """The recommender works only through new SKUs: controlling for them leaves ~nothing."""
    full = d["reps_full"]
    assert north_coef(full, "new_skus ~ north + relationship_strength + experience") > 2
    f = "sales_post ~ north + " + X + " + relationship_strength"
    for m in [d] + many:
        tr = m["truth"]["att_north_rep_abs"]
        assert north_coef(m["reps_full"], f + " + new_skus") < 0.35 * tr


def test_collider(d):
    full = d["reps_full"]
    rate = full.groupby("north").bonus.mean()
    assert 0.5 < rate[1] < 0.9 and 0.03 < rate[0] < 0.3   # both groups have both values
    gr = full.sales_post / full.sales_pre
    assert np.corrcoef(gr, full.bonus)[0, 1] > 0.2
    f = "sales_post ~ north + " + X + " + relationship_strength"
    assert north_coef(full, f + " + bonus") < north_coef(full, f) - 0.1   # conditioning biases down


# --- effect, estimands -----------------------------------------------------------

def test_estimands(d, many):
    t = d["truth"]
    assert t["att_north_store_pct"] < t["ate_rep_pct"]   # North reps are more experienced
    assert 0.07 < t["att_north_store_pct"] < 0.12
    assert t["rollout_spillover_outside_north"] == 0.0
    assert t["cate_abs_slope_per_year"] < 0
    assert all(m["truth"]["att_north_store_pct"] < m["truth"]["ate_rep_pct"] for m in many)


def test_rep_truth_matches_truth(d):
    rt, t = d["rep_truth"], d["truth"]
    assert rt.true_effect_abs.mean() == pytest.approx(t["ate_rep_abs"], rel=1e-3)
    assert rt.true_effect_abs[rt.north == 1].mean() == pytest.approx(t["att_north_rep_abs"], rel=1e-3)


def test_cate_function():
    p = ud.Params()
    e = np.array([2.0, 6.5, 12.0, 30.0])
    c = ud.cate_pct(e, p)
    assert np.all(np.diff(c) <= 0)
    assert c[1] == pytest.approx(p.effect_mean)
    assert c[-1] == pytest.approx(p.effect_mean * p.effect_floor)


def test_effect_matches_cate_on_average(d):
    w = d["world"]
    implied = w.store_pct / ud.effect_multiplier(w.rep_experience[w.store_rep], w.p)
    assert implied.mean() == pytest.approx(w.p.effect_mean, rel=0.02)


# --- interference ----------------------------------------------------------------

def test_spillover_by_hand():
    p = ud.Params(reps_per_region=2, stores_min=4, stores_max=4, n_ab_reps=4)
    w = ud.World(p)
    tr = np.zeros(w.mu0.shape, bool)
    tr[[0, 1], 5] = True                           # 2 of the 4 stores of rep 0 (North), week 5
    _, tau = ud.simulate(w, tr)
    share = tau[:, 5] / (w.mu0[:, 5] * w.store_pct)
    assert np.allclose(share[[0, 1]], 1.0)         # treated
    assert np.allclose(share[[2, 3]], 0.25)        # same rep: 0.5 x half the stores
    assert np.allclose(share[4:8], 0.10)           # other rep, same region: 0.1 x 1 of 1 other rep
    assert np.allclose(share[8:], 0.0)             # other regions
    assert np.all(tau[:, 4] == 0) and np.all(tau[:, 6] == 0)


def test_rep_randomized_ab(d):
    t, ab = d["truth"], d["ab_rep"]
    assert t["ab_rep_control_spill_abs"] < 0.05 * t["ab_rep_direct_abs"]
    assert set(ab.rep_treated_share.unique()) <= {0.0, 1.0}
    m = smf.ols("sales_test ~ treated + sales_pre", ab).fit(cov_type="cluster", cov_kwds={"groups": ab.rep_id})
    lo, hi = m.conf_int().loc["treated"]
    assert lo < t["ab_rep_expected_diff_abs"] < hi
    assert t["ab_rep_expected_diff_pct"] < t["ab_rep_direct_pct"] + 0.01


def test_store_randomized_ab_is_attenuated(d):
    t, ab = d["truth"], d["ab_store"]
    share = t["ab_store_control_spill_abs"] / t["ab_store_direct_abs"]
    assert 0.18 < share < 0.32                     # gamma_within x ~half the rep's stores
    assert ab.rep_treated_share.between(0.2, 0.8).all()
    est = smf.ols("sales_test ~ treated + sales_pre", ab).fit().params["treated"]
    assert est < 0.85 * t["ab_store_direct_abs"]
    assert est == pytest.approx(t["ab_store_expected_diff_abs"], rel=0.2)


# --- synthetic control -----------------------------------------------------------

def test_sc_fits_and_recovers(d, many):
    p = d["world"].p
    gap = synth(d["region_week"], p.n_pre)
    rw = d["region_week"]
    north_mean = rw[(rw.region == "North") & (rw.week < p.n_pre)].sales.mean()
    assert np.sqrt((gap[:p.n_pre] ** 2).mean()) < 0.02 * north_mean
    tr = d["truth"]["att_north_store_abs"]
    assert abs(sc_att(d) - tr) < 0.15 * tr
    assert abs(sc_att(d, intercept=False) - tr) < 0.2 * tr
    err = np.array([sc_att(m) / m["truth"]["att_north_store_abs"] - 1 for m in many])
    obs = np.array([dml_obs(m) / m["truth"]["att_north_rep_abs"] - 1 for m in many])
    assert abs(err.mean()) < 0.12
    assert np.mean(np.abs(err) < np.abs(obs)) >= 0.7   # SC closer than observed DML, mostly


def test_sc_fake_date_shows_no_gap(d, many):
    """No gap on average. A single world can show some gap: that is the check doing its job
    (SC's own bias shows up before the real date too)."""
    tr = d["truth"]["att_north_store_abs"]
    assert abs(fake_date_gap(d)) < 0.2 * tr
    assert abs(np.mean([fake_date_gap(m) for m in many])) < 0.1 * tr


def test_sc_in_space_placebo_ranks_north_first(d):
    p = d["world"].p
    rw = d["region_week"]
    ratios = {}
    for r in ud.REGIONS:
        # a fake North takes the North's place; the real North is never a donor
        df = rw if r == "North" else rw[rw.region != "North"].replace({"region": {r: "North"}})
        gap = synth(df, p.n_pre)
        ratios[r] = np.sqrt((gap[p.n_pre:] ** 2).mean()) / np.sqrt((gap[:p.n_pre] ** 2).mean())
    assert max(ratios, key=ratios.get) == "North"


def test_hull_seed_breaks_sc():
    """The notebook 04 world: the North is outside the hull, SC is off and the fake date says so."""
    h = ud.make_datasets(ud.Params(seed=ud.HULL_SEED))
    tr = h["truth"]["att_north_store_abs"]
    assert h["world"].north_mix_miss > 0.5
    assert sc_att(h) - tr > 0.15 * tr
    assert abs(fake_date_gap(h)) > 0.15 * tr
    # SC that also allows a linear trend gap: close to the truth, and the fake date is quiet
    n = P(h).n_pre
    assert abs(synth_trend(h["region_week"], n)[n:].mean() - tr) < 0.12 * tr
    rw = h["region_week"][h["region_week"].week < n]
    assert abs(synth_trend(rw, n - 20)[n - 20:].mean()) < 0.05 * tr
