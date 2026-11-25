"""Agreement between engine counts and manual (reference) counts.

Implements the validation protocol from the technical research method:
ICC(2,1) absolute agreement, normalised mean absolute error, Bland-Altman limits of
agreement, and a check that errors do not differ between glasses conditions.
Confidence intervals use a bootstrap over clips (2,000 resamples, fixed seed).
"""
import numpy as np
import pandas as pd

# Proposed acceptance thresholds (agree these with the team before fieldwork)
THRESHOLDS = {
    "icc_lower_min": 0.75,      # lower 95% bound of ICC for "use in analysis"
    "icc_lower_desc": 0.50,     # lower 95% bound of ICC for "descriptive only"
    "nmae_max": 10.0,           # % normalised mean absolute error for "use in analysis"
    "nmae_desc": 20.0,
    "loa_max_pct": 15.0,        # Bland-Altman limits, as % of mean count
    "condition_gap_max": 5.0,   # max difference in mean % error between conditions
}


def icc_a1(x):
    """ICC(2,1) / ICC(A,1): two-way random effects, absolute agreement, single rater.
    x: array (n_clips, k_raters). Returns np.nan if not computable."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 3 or k < 2:
        return np.nan
    gm = x.mean()
    ss_r = k * ((x.mean(axis=1) - gm) ** 2).sum()
    ss_c = n * ((x.mean(axis=0) - gm) ** 2).sum()
    ss_t = ((x - gm) ** 2).sum()
    ss_e = ss_t - ss_r - ss_c
    ms_r, ms_c, ms_e = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom > 0 else np.nan


def _boot(fn, n, reps=2000, seed=7):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < 50:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def agreement(engine_counts, reference_counts, conditions=None):
    e = np.asarray(engine_counts, dtype=float)
    r = np.asarray(reference_counts, dtype=float)
    ok = ~(np.isnan(e) | np.isnan(r)) & (r > 0)
    e, r = e[ok], r[ok]
    cond = np.asarray(conditions)[ok] if conditions is not None else None
    n = len(e)
    out = {"n_clips": n}
    if n == 0:
        return out
    diff = e - r
    pct = diff / r * 100.0
    out.update({
        "mae": round(float(np.mean(np.abs(diff))), 2),
        "nmae_pct": round(float(np.mean(np.abs(pct))), 1),
        "within_5pct": round(float(np.mean(np.abs(pct) <= 5) * 100), 1),
        "within_10pct": round(float(np.mean(np.abs(pct) <= 10) * 100), 1),
        "bias": round(float(np.mean(diff)), 2),
        "bias_pct": round(float(np.mean(pct)), 1),
    })
    if n >= 2:
        sd = float(np.std(diff, ddof=1))
        out["loa_low"], out["loa_high"] = round(out["bias"] - 1.96 * sd, 2), round(out["bias"] + 1.96 * sd, 2)
        sdp = float(np.std(pct, ddof=1))
        out["loa_low_pct"], out["loa_high_pct"] = round(out["bias_pct"] - 1.96 * sdp, 1), round(out["bias_pct"] + 1.96 * sdp, 1)
    pair = np.column_stack([e, r])
    out["icc"] = icc_a1(pair)
    lo, hi = _boot(lambda idx: icc_a1(pair[idx]), n) if n >= 5 else (np.nan, np.nan)
    out["icc_ci_low"], out["icc_ci_high"] = lo, hi
    lo, hi = _boot(lambda idx: float(np.mean(np.abs(pct[idx]))), n) if n >= 5 else (np.nan, np.nan)
    out["nmae_ci_low"], out["nmae_ci_high"] = lo, hi

    if cond is not None:
        labels = [c for c in pd.unique(cond) if str(c) not in ("", "nan", "Not applicable")]
        if len(labels) == 2:
            a, b = (cond == labels[0]), (cond == labels[1])
            if a.sum() >= 2 and b.sum() >= 2:
                gap = float(np.mean(pct[a]) - np.mean(pct[b]))
                out["condition_gap_pct"] = round(gap, 1)
                out["condition_gap_label"] = f"{labels[0]} minus {labels[1]}"

                def gfn(idx):
                    ia, ib = idx[a[idx]], idx[b[idx]]
                    return float(np.mean(pct[ia]) - np.mean(pct[ib])) if len(ia) and len(ib) else np.nan
                out["condition_gap_ci_low"], out["condition_gap_ci_high"] = _boot(gfn, n)
    out["verdict"], out["verdict_reasons"] = verdict(out)
    return out


def verdict(a, th=THRESHOLDS):
    reasons = []
    if a.get("n_clips", 0) < 10:
        return "too few clips", ["Fewer than 10 clips with a reference count."]
    icc_lo = a.get("icc_ci_low", np.nan)
    nmae = a.get("nmae_pct", np.nan)
    loa = max(abs(a.get("loa_low_pct", np.inf)), abs(a.get("loa_high_pct", np.inf)))
    gap = a.get("condition_gap_pct")
    gap_lo, gap_hi = a.get("condition_gap_ci_low", np.nan), a.get("condition_gap_ci_high", np.nan)
    excludes_zero = (gap_lo > 0 or gap_hi < 0)
    cond_fail = gap is not None and abs(gap) > th["condition_gap_max"] and excludes_zero
    cond_unsure = gap is not None and abs(gap) > th["condition_gap_max"] and not excludes_zero
    cond_ok = not (cond_fail or cond_unsure)
    if cond_fail:
        reasons.append(f"Error differs between conditions ({gap:+.1f} points, clearly not chance): "
                       "this would bias the glasses comparison.")
    elif cond_unsure:
        reasons.append(f"Error differs between conditions by {gap:+.1f} points; with these clips it may be chance, "
                       "but it cannot be ruled out. Add clips before relying on this engine.")
    if np.isnan(icc_lo) or icc_lo < th["icc_lower_desc"] or nmae > th["nmae_desc"]:
        reasons.append(f"ICC lower bound {icc_lo:.2f} / error {nmae:.1f}% too weak.")
        return "not usable", reasons
    if cond_fail:
        return "not usable", reasons
    if icc_lo >= th["icc_lower_min"] and nmae <= th["nmae_max"] and loa <= th["loa_max_pct"] and cond_ok:
        return "meets thresholds", ["All proposed thresholds met."]
    if icc_lo < th["icc_lower_min"]:
        reasons.append(f"ICC lower bound {icc_lo:.2f} below {th['icc_lower_min']}.")
    if nmae > th["nmae_max"]:
        reasons.append(f"Normalised error {nmae:.1f}% above {th['nmae_max']}%.")
    if loa > th["loa_max_pct"]:
        reasons.append(f"Limits of agreement reach ±{loa:.0f}% (target ±{th['loa_max_pct']:.0f}%).")
    return ("not usable" if cond_fail else "descriptive only"), reasons


def inter_rater(c1, c2):
    x = np.column_stack([np.asarray(c1, float), np.asarray(c2, float)])
    x = x[~np.isnan(x).any(axis=1)]
    if len(x) < 3:
        return {"n_clips": len(x)}
    lo, hi = _boot(lambda idx: icc_a1(x[idx]), len(x)) if len(x) >= 5 else (np.nan, np.nan)
    pct = (x[:, 0] - x[:, 1]) / np.maximum(x.mean(axis=1), 1e-9) * 100
    return {"n_clips": len(x), "icc": icc_a1(x), "icc_ci_low": lo, "icc_ci_high": hi,
            "mean_abs_diff_pct": round(float(np.mean(np.abs(pct))), 1)}


MANUAL_COLUMNS = ["clip_label", "video_file", "start_sec", "end_sec", "coder1_count", "coder2_count",
                  "artisan_id", "condition"]


def manual_template():
    return pd.DataFrame([{c: ("" if c not in ("start_sec", "end_sec") else 0.0) for c in MANUAL_COLUMNS}]).iloc[0:0]
