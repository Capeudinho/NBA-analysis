"""Outlier detection methods.

Univariate (applied column by column):
    Z-score, robust (modified) Z-score, IQR/Tukey, Grubbs, ECOD.
Multivariate (applied to each game as a point in the space of the analysis variables):
    ECOD, K-NN distance, LOF and DBSCAN.

Score-based methods (ECOD, K-NN, LOF) produce a continuous anomaly score; the
binary decision uses the same Tukey rule adopted in the rest of the report,
applied to the scores: score > Q3(score) + 1.5 x IQR(score).
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from . import config
from .data import numeric_quality_columns, team_data

UNIVARIATE_METHODS = {
    "zscore": "Z-score (|z| > 3)",
    "robust_z": "Z-score robusto (MAD, |z*| > 3,5)",
    "iqr": "IQR / Boxplot",
    "grubbs": "Grubbs (α = 0,05)",
    "ecod": "ECOD",
}
MULTIVARIATE_METHODS = {
    "ecod": "ECOD multivariado",
    "knn": f"K-NN (k = {config.KNN_K})",
    "lof": f"LOF (k = {config.LOF_NEIGHBORS})",
    "dbscan": "DBSCAN",
}
ROBUST_Z_THRESHOLD = 3.5


def score_threshold(scores, k=config.IQR_K):
    """Tukey's upper fence applied to anomaly scores."""
    q1, q3 = np.percentile(scores, [25, 75])
    return float(q3 + k * (q3 - q1))


# ----------------------------------------------------------------------------
# Univariate methods: each returns a boolean mask plus diagnostic information.
# ----------------------------------------------------------------------------

def zscore_outliers(x, threshold=config.Z_THRESHOLD):
    x = np.asarray(x, dtype=float)
    std = x.std(ddof=1)
    if std == 0:
        return np.zeros(x.size, dtype=bool), {"threshold": threshold}
    z = (x - x.mean()) / std
    return np.abs(z) > threshold, {"threshold": threshold, "scores": z}


def robust_zscore_outliers(x, threshold=ROBUST_Z_THRESHOLD):
    """Iglewicz & Hoaglin modified z-score: 0.6745 (x - median) / MAD."""
    x = np.asarray(x, dtype=float)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad > 0:
        z = 0.6745 * (x - median) / mad
    else:
        mean_ad = np.mean(np.abs(x - median))
        if mean_ad == 0:
            return np.zeros(x.size, dtype=bool), {"threshold": threshold}
        z = (x - median) / (1.253314 * mean_ad)
    return np.abs(z) > threshold, {"threshold": threshold, "scores": z}


def iqr_outliers(x, k=config.IQR_K):
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (x < lower) | (x > upper), {"lower": lower, "upper": upper}


def grubbs_critical_value(n, alpha=config.GRUBBS_ALPHA):
    t = sps.t.ppf(1 - alpha / (2 * n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))


def grubbs_outliers(x, alpha=config.GRUBBS_ALPHA, max_fraction=0.1):
    """Two-sided Grubbs test applied iteratively (removing one point at a time)."""
    x = np.asarray(x, dtype=float)
    mask = np.zeros(x.size, dtype=bool)
    remaining = np.arange(x.size)
    steps = []
    max_removals = max(1, int(max_fraction * x.size))
    while remaining.size > 2 and len(steps) < max_removals:
        values = x[remaining]
        std = values.std(ddof=1)
        if std == 0:
            break
        deviations = np.abs(values - values.mean())
        position = int(np.argmax(deviations))
        g_stat = deviations[position] / std
        g_crit = grubbs_critical_value(values.size, alpha)
        steps.append({"value": float(values[position]), "G": float(g_stat), "G_crit": float(g_crit)})
        if g_stat <= g_crit:
            break
        mask[remaining[position]] = True
        remaining = np.delete(remaining, position)
    return mask, {"steps": steps, "alpha": alpha}


def ecod_tail_scores(X):
    """Per-dimension ECOD scores: -log of the left/right empirical tail probabilities."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1:
        X = X.T
    n = X.shape[0]
    left = np.empty_like(X)
    right = np.empty_like(X)
    for j in range(X.shape[1]):
        column = X[:, j]
        ordered = np.sort(column)
        left[:, j] = np.searchsorted(ordered, column, side="right") / n  # P(X <= x)
        right[:, j] = (n - np.searchsorted(ordered, column, side="left")) / n  # P(X >= x)
    return -np.log(left), -np.log(right)


def ecod_scores(X):
    """ECOD (Li et al., 2022): O = max(O_left, O_right, O_auto)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1:
        X = X.T
    neg_log_left, neg_log_right = ecod_tail_scores(X)
    skew = sps.skew(X, axis=0)
    auto = np.where(skew < 0, neg_log_left, neg_log_right)
    o_left = neg_log_left.sum(axis=1)
    o_right = neg_log_right.sum(axis=1)
    o_auto = auto.sum(axis=1)
    per_dimension = np.maximum(neg_log_left, neg_log_right)
    return np.maximum.reduce([o_left, o_right, o_auto]), per_dimension


def ecod_outliers(x, k=config.IQR_K):
    scores, _ = ecod_scores(np.asarray(x, dtype=float).reshape(-1, 1))
    threshold = score_threshold(scores, k)
    return scores > threshold, {"threshold": threshold, "scores": scores}


UNIVARIATE_FUNCTIONS = {
    "zscore": zscore_outliers,
    "robust_z": robust_zscore_outliers,
    "iqr": iqr_outliers,
    "grubbs": grubbs_outliers,
    "ecod": ecod_outliers,
}


def _run_univariate(key, x, k):
    function = UNIVARIATE_FUNCTIONS[key]
    if key in ("iqr", "ecod"):
        return function(x, k=k)
    return function(x)


@lru_cache(maxsize=None)
def univariate_table(scope, k=config.IQR_K):
    """Outlier counts per column (rows) and method (columns) for a scope."""
    df = team_data(scope)
    rows = []
    for column in numeric_quality_columns(df):
        x = df[column].to_numpy(dtype=float)
        counts = {}
        for key in UNIVARIATE_FUNCTIONS:
            mask, _ = _run_univariate(key, x, k)
            counts[key] = int(mask.sum())
        rows.append({"column": column, "n": int(x.size), **counts})
    return rows


def univariate_detail(scope, column, k=config.IQR_K):
    """Per-game flags for a single column (used by the detail chart and table)."""
    df = team_data(scope)
    x = df[column].to_numpy(dtype=float)
    flags = {}
    info = {}
    for key in UNIVARIATE_FUNCTIONS:
        mask, details = _run_univariate(key, x, k)
        flags[key] = mask
        info[key] = details
    flag_frame = pd.DataFrame(flags, index=df.index)
    detail = df[["game_date", "team", "matchup", column]].copy()
    detail["n_methods"] = flag_frame.sum(axis=1)
    for key in flags:
        detail[key] = flag_frame[key]
    return detail, info


# ----------------------------------------------------------------------------
# Multivariate methods on the standardized analysis variables
# ----------------------------------------------------------------------------

def knee_point(sorted_values):
    """Point of maximum distance between a sorted curve and its chord (Kneedle idea)."""
    y = np.asarray(sorted_values, dtype=float)
    n = y.size
    if n < 3 or y[-1] == y[0]:
        return float(y[-1])
    x_norm = np.linspace(0, 1, n)
    y_norm = (y - y[0]) / (y[-1] - y[0])
    gap = x_norm - y_norm  # chord (y = x) minus convex curve
    return float(y[int(np.argmax(gap))])


@lru_cache(maxsize=None)
def multivariate_analysis(scope, k=config.IQR_K):
    df = team_data(scope)
    variables = config.VARIABLE_KEYS
    values = df[variables].to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1)
    Z = (values - means) / stds
    n, d = Z.shape

    results = {}

    ecod, ecod_dims = ecod_scores(Z)
    results["ecod"] = {"scores": ecod, "threshold": score_threshold(ecod, k)}

    n_neighbors = min(config.KNN_K, n - 1)
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Z).kneighbors(Z)
    knn = distances[:, 1:].mean(axis=1)
    results["knn"] = {"scores": knn, "threshold": score_threshold(knn, k)}

    lof_model = LocalOutlierFactor(n_neighbors=min(config.LOF_NEIGHBORS, n - 1))
    lof_model.fit(Z)
    lof = -lof_model.negative_outlier_factor_
    results["lof"] = {"scores": lof, "threshold": score_threshold(lof, k)}

    min_samples = 2 * d
    k_dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)
    k_distances = np.sort(k_dist[:, -1])
    eps = knee_point(k_distances)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(Z).labels_
    results["dbscan"] = {
        "labels": labels,
        "eps": eps,
        "min_samples": min_samples,
        "k_distances": k_distances,
        "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
    }

    flags = pd.DataFrame(index=df.index)
    for key in ("ecod", "knn", "lof"):
        flags[key] = results[key]["scores"] > results[key]["threshold"]
    flags["dbscan"] = labels == -1

    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    coordinates = pca.fit_transform(Z)

    games = df[["game_date", "team", "matchup", "win"] + variables].copy()
    for key in flags:
        games[f"flag_{key}"] = flags[key].to_numpy()
    games["n_methods"] = flags.sum(axis=1).to_numpy()
    games["ecod_score"] = ecod
    games["knn_score"] = knn
    games["lof_score"] = lof
    games["pc1"] = coordinates[:, 0]
    games["pc2"] = coordinates[:, 1]

    # Attribution: which variables are individually extreme (|z| >= 2) in flagged games.
    z_frame = pd.DataFrame(Z, columns=variables, index=df.index)
    attribution = {}
    for key in flags:
        flagged = z_frame[flags[key]]
        attribution[key] = {var: int((flagged[var].abs() >= 2).sum()) for var in variables}
        attribution[key]["none"] = int((flagged.abs() < 2).all(axis=1).sum())

    games["drivers"] = [
        [var for var in variables if abs(z_frame.at[idx, var]) >= 2] for idx in df.index
    ]

    return {
        "n": n,
        "d": d,
        "results": results,
        "games": games,
        "counts": {key: int(flags[key].sum()) for key in flags},
        "attribution": attribution,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "agreement": _agreement_matrix(flags),
        "ecod_dimensions": ecod_dims,
    }


def _agreement_matrix(flags):
    keys = list(flags.columns)
    matrix = {}
    for a in keys:
        matrix[a] = {}
        for b in keys:
            both = int((flags[a] & flags[b]).sum())
            either = int((flags[a] | flags[b]).sum())
            matrix[a][b] = both / either if either else 1.0
    return matrix
