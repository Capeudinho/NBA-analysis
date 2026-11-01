"""Univariate descriptive statistics: central tendency, dispersion, skewness, CV, IQR.

Conventions (kept identical across every section of the report):
- Standard deviation and variance are *sample* estimates (denominator n - 1).
- Quartiles use linear interpolation (Hyndman & Fan type 7, the numpy/pandas default).
- Outlier fences follow Tukey's rule: Q1 - k*IQR and Q3 + k*IQR with k = 1.5.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config
from .data import get_dataset, team_data


def _array(values):
    return np.asarray(values, dtype=float)


def quartiles(values):
    x = _array(values)
    q1, median, q3 = np.percentile(x, [25, 50, 75])  # linear interpolation (type 7)
    return float(q1), float(median), float(q3)


def modes(values):
    """All modal values (a variable can be multimodal)."""
    series = pd.Series(_array(values))
    counts = series.value_counts()
    top = counts.max()
    return sorted(float(v) for v in counts[counts == top].index), int(top)


def tukey_fences(values, k=config.IQR_K):
    q1, _, q3 = quartiles(values)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def describe(values, k=config.IQR_K):
    """Central tendency, dispersion, quartiles and Tukey outliers of a sample."""
    x = _array(values)
    q1, median, q3 = quartiles(x)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mode_values, mode_count = modes(x)
    inside = x[(x >= lower) & (x <= upper)]
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": median,
        "modes": mode_values,
        "mode": mode_values[0],
        "mode_count": mode_count,
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "std": float(x.std(ddof=1)),
        "var": float(x.var(ddof=1)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "whisker_low": float(inside.min()) if inside.size else lower,
        "whisker_high": float(inside.max()) if inside.size else upper,
        "outliers_low": sorted(float(v) for v in x[x < lower]),
        "outliers_high": sorted(float(v) for v in x[x > upper]),
        "n_outliers": int(((x < lower) | (x > upper)).sum()),
    }


def outlier_games(df, variable, k=config.IQR_K):
    """Games (rows) whose value of `variable` falls outside Tukey's fences."""
    lower, upper = tukey_fences(df[variable], k)
    mask = (df[variable] < lower) | (df[variable] > upper)
    rows = df.loc[mask, ["game_date", "matchup", "opponent", "is_home", "win", variable]]
    records = []
    for _, row in rows.sort_values(variable, ascending=False).iterrows():
        records.append(
            {
                "date": row["game_date"],
                "matchup": row["matchup"],
                "opponent": row["opponent"],
                "is_home": int(row["is_home"]),
                "win": int(row["win"]),
                "value": float(row[variable]),
                "side": "superior" if row[variable] > upper else "inferior",
            }
        )
    return records


# ----------------------------------------------------------------------------
# Skewness
# ----------------------------------------------------------------------------

def skewness_standard_error(n):
    """Standard error of the adjusted Fisher-Pearson coefficient G1."""
    return float(np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))


def classify_direction(g):
    if g > 0:
        return "positiva (à direita)"
    if g < 0:
        return "negativa (à esquerda)"
    return "simétrica"


def classify_intensity(g):
    magnitude = abs(g)
    if magnitude < config.SKEW_MODERATE:
        return "aproximadamente simétrica"
    if magnitude < config.SKEW_STRONG:
        return "moderada"
    return "forte"


def skewness(values):
    """Skewness via three formulas. G1 (adjusted Fisher-Pearson) is the reference."""
    x = _array(values)
    n = x.size
    g1 = float(sps.skew(x, bias=True))  # Pearson moment coefficient m3 / m2^1.5
    big_g1 = float(sps.skew(x, bias=False))  # adjusted Fisher-Pearson (pandas .skew())
    q1, median, q3 = quartiles(x)
    bowley = float((q3 + q1 - 2 * median) / (q3 - q1)) if q3 > q1 else 0.0
    ses = skewness_standard_error(n)
    z = big_g1 / ses
    mean = float(x.mean())
    return {
        "n": n,
        "g1": g1,
        "G1": big_g1,
        "bowley": bowley,
        "ses": ses,
        "z": z,
        "p_value": float(2 * (1 - sps.norm.cdf(abs(z)))),
        "significant": abs(z) > sps.norm.ppf(1 - config.SIGNIFICANCE / 2),
        "direction": classify_direction(big_g1),
        "intensity": classify_intensity(big_g1),
        "mean": mean,
        "median": median,
        "mean_minus_median": mean - median,
        "bowley_direction": classify_direction(bowley) if abs(bowley) > 1e-12 else "simétrica",
    }


# ----------------------------------------------------------------------------
# Coefficient of variation
# ----------------------------------------------------------------------------

def coefficient_of_variation(values):
    """CV = s / mean x 100 (sample s, result in percent) plus robust alternatives."""
    x = _array(values)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    q1, median, q3 = quartiles(x)
    mad = float(np.median(np.abs(x - median)))
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "cv": std / mean * 100 if mean else np.nan,
        "cv_mad": 1.4826 * mad / median * 100 if median else np.nan,
        "cv_iqr": (q3 - q1) / median * 100 if median else np.nan,
        "cv_poisson": 100 / np.sqrt(mean) if mean > 0 else np.nan,
        "dispersion_index": float(x.var(ddof=1)) / mean if mean else np.nan,
        "mean_over_std": mean / std if std else np.nan,
        "small_mean": mean < config.CV_SMALL_MEAN,
        "ratio_scale": True,
        "mean_positive": mean > 0,
        "zeros": int((x == 0).sum()),
    }


def classify_cv(cv):
    if cv < 15:
        return "baixa dispersão"
    if cv < 30:
        return "dispersão moderada"
    return "alta dispersão"


# ----------------------------------------------------------------------------
# Interquartile range
# ----------------------------------------------------------------------------

def iqr_analysis(values):
    x = _array(values)
    d = describe(x)
    iqr = d["iqr"]
    result = {
        "q1": d["q1"],
        "q3": d["q3"],
        "iqr": iqr,
        "median": d["median"],
        "range": d["range"],
        "std": d["std"],
        "range_over_iqr": d["range"] / iqr if iqr else np.nan,
        "std_over_iqr": d["std"] / iqr if iqr else np.nan,
        # Under normality IQR = 1.349 sigma, so sigma / IQR = 0.741.
        "normal_std_over_iqr": 1 / (2 * sps.norm.ppf(0.75)),
    }
    for label, k in (("k15", config.IQR_K), ("k30", config.IQR_K_EXTREME)):
        lower, upper = d["q1"] - k * iqr, d["q3"] + k * iqr
        mask = (x < lower) | (x > upper)
        result[label] = {
            "k": k,
            "lower": lower,
            "upper": upper,
            "n_low": int((x < lower).sum()),
            "n_high": int((x > upper).sum()),
            "n": int(mask.sum()),
        }
    inside = x[(x >= result["k15"]["lower"]) & (x <= result["k15"]["upper"])]
    result["std_without_outliers"] = float(inside.std(ddof=1)) if inside.size > 1 else np.nan
    q1_in, _, q3_in = quartiles(inside) if inside.size else (np.nan, np.nan, np.nan)
    result["iqr_without_outliers"] = q3_in - q1_in
    result["std_change_pct"] = (
        (d["std"] - result["std_without_outliers"]) / d["std"] * 100 if d["std"] else 0.0
    )
    result["iqr_change_pct"] = (
        (iqr - result["iqr_without_outliers"]) / iqr * 100 if iqr else 0.0
    )
    # Consistency check: the outliers found here are the ones drawn in the boxplot.
    boxplot_outliers = sorted(d["outliers_low"] + d["outliers_high"])
    recomputed = sorted(float(v) for v in x[(x < result["k15"]["lower"]) | (x > result["k15"]["upper"])])
    result["matches_boxplot"] = boxplot_outliers == recomputed
    return result


# ----------------------------------------------------------------------------
# Cached tables for the group's teams
# ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def team_variable_summary():
    """Nested dict: summary[variable][team] -> all measures for the group's teams."""
    summary = {}
    for variable in config.VARIABLE_KEYS:
        summary[variable] = {}
        for team in config.TEAM_CODES:
            values = team_data(team)[variable]
            summary[variable][team] = {
                "describe": describe(values),
                "skew": skewness(values),
                "cv": coefficient_of_variation(values),
                "iqr": iqr_analysis(values),
            }
    return summary


@lru_cache(maxsize=None)
def league_reference():
    """League benchmark: game-level stats and the distribution of the 30 team means."""
    df = get_dataset()
    reference = {}
    for variable in config.VARIABLE_KEYS:
        team_means = df.groupby("team")[variable].mean()
        reference[variable] = {
            "game_mean": float(df[variable].mean()),
            "game_std": float(df[variable].std(ddof=1)),
            "team_means_mean": float(team_means.mean()),
            "team_means_std": float(team_means.std(ddof=1)),
            "team_means": team_means.to_dict(),
            "rank": {
                team: int(team_means.rank(ascending=False, method="min")[team])
                for team in config.TEAM_CODES
            },
        }
    return reference


def team_mean_zscores():
    """z = (team mean - mean of the 30 team means) / std of the 30 team means."""
    reference = league_reference()
    z = {}
    for variable in config.VARIABLE_KEYS:
        ref = reference[variable]
        z[variable] = {
            team: (ref["team_means"][team] - ref["team_means_mean"]) / ref["team_means_std"]
            for team in config.TEAM_CODES
        }
    return z
