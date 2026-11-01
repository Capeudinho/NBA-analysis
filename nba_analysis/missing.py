"""Missing-data analysis: per-column counts, MCAR tests and a simulation lab.

Tests implemented from scratch (numpy/scipy):
- Little's MCAR test (Little, 1988), with EM estimates of mean and covariance.
- Jamshidian & Jalal (2010) test: normal-based imputation + Hawkins test of
  homogeneity of covariances across missing-data patterns, followed by a
  nonparametric k-sample Anderson-Darling test when Hawkins rejects.
- Listing & Schlittgen (LS) nonparametric test, cross-sectional version:
  Wilcoxon-Mann-Whitney comparisons between cases with and without missing values.
- Fairclough's logistic regression of the missingness indicator on the observed data.
"""

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config
from .data import get_dataset, load_raw

MECHANISMS = {
    "MCAR": "MCAR — ausência completamente aleatória",
    "MAR": "MAR — ausência depende de outra variável observada",
    "MNAR": "MNAR — ausência depende do próprio valor ausente",
}
RATES = [0.05, 0.10, 0.20, 0.30]
DEPENDENCE_STRENGTH = 1.5
MIN_GROUP_SIZE = 6


def missing_summary():
    raw = load_raw()
    total = len(raw)
    rows = []
    for column in raw.columns:
        n_missing = int(raw[column].isna().sum())
        blank = 0
        if raw[column].dtype == object or pd.api.types.is_string_dtype(raw[column]):
            blank = int(raw[column].astype(str).str.strip().isin(["", "NA", "N/A", "null", "None", "-"]).sum())
        rows.append(
            {
                "column": column,
                "missing": n_missing,
                "blank": blank,
                "pct": n_missing / total * 100,
            }
        )
    patterns = raw.isna().drop_duplicates().shape[0]
    return rows, patterns


# ----------------------------------------------------------------------------
# EM algorithm for the multivariate normal with missing values
# ----------------------------------------------------------------------------

def _patterns(X):
    observed = ~np.isnan(X)
    keys, inverse = np.unique(observed, axis=0, return_inverse=True)
    return keys, inverse.ravel()


def em_mvn(X, max_iter=500, tol=1e-8):
    n, p = X.shape
    keys, inverse = _patterns(X)
    mu = np.nanmean(X, axis=0)
    sigma = np.diag(np.nanvar(X, axis=0))
    for iteration in range(1, max_iter + 1):
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))
        for g, observed in enumerate(keys):
            rows = X[inverse == g]
            o = np.where(observed)[0]
            m = np.where(~observed)[0]
            filled = rows.copy()
            if m.size:
                coef = sigma[np.ix_(m, o)] @ np.linalg.inv(sigma[np.ix_(o, o)])
                filled[:, m] = mu[m] + (rows[:, o] - mu[o]) @ coef.T
                conditional_cov = sigma[np.ix_(m, m)] - coef @ sigma[np.ix_(o, m)]
                sum_xx[np.ix_(m, m)] += rows.shape[0] * conditional_cov
            sum_x += filled.sum(axis=0)
            sum_xx += filled.T @ filled
        new_mu = sum_x / n
        new_sigma = sum_xx / n - np.outer(new_mu, new_mu)
        converged = np.max(np.abs(new_mu - mu)) < tol and np.max(np.abs(new_sigma - sigma)) < tol
        mu, sigma = new_mu, new_sigma
        if converged:
            break
    return mu, sigma, iteration


# ----------------------------------------------------------------------------
# Little's MCAR test
# ----------------------------------------------------------------------------

def little_mcar_test(X):
    n, p = X.shape
    keys, inverse = _patterns(X)
    if len(keys) < 2:
        return {"applicable": False}
    mu, sigma, iterations = em_mvn(X)
    d2 = 0.0
    df = 0
    for g, observed in enumerate(keys):
        o = np.where(observed)[0]
        if o.size == 0:
            continue
        rows = X[inverse == g][:, o]
        diff = rows.mean(axis=0) - mu[o]
        d2 += rows.shape[0] * diff @ np.linalg.inv(sigma[np.ix_(o, o)]) @ diff
        df += o.size
    df -= p
    return {
        "applicable": True,
        "statistic": float(d2),
        "df": int(df),
        "p_value": float(sps.chi2.sf(d2, df)),
        "patterns": int(len(keys)),
        "em_iterations": iterations,
    }


# ----------------------------------------------------------------------------
# Jamshidian & Jalal test
# ----------------------------------------------------------------------------

def normal_imputation(X, mu, sigma, rng):
    """Impute each missing block from its conditional normal distribution."""
    keys, inverse = _patterns(X)
    imputed = X.copy()
    for g, observed in enumerate(keys):
        m = np.where(~observed)[0]
        if not m.size:
            continue
        o = np.where(observed)[0]
        index = np.where(inverse == g)[0]
        rows = X[index]
        coef = sigma[np.ix_(m, o)] @ np.linalg.inv(sigma[np.ix_(o, o)])
        mean = mu[m] + (rows[:, o] - mu[o]) @ coef.T
        cov = sigma[np.ix_(m, m)] - coef @ sigma[np.ix_(o, m)]
        noise = rng.multivariate_normal(np.zeros(m.size), cov, size=index.size)
        imputed[np.ix_(index, m)] = mean + noise
    return imputed


LEGENDRE = [
    lambda u: np.sqrt(3) * (2 * u - 1),
    lambda u: np.sqrt(5) * (6 * u**2 - 6 * u + 1),
    lambda u: np.sqrt(7) * (20 * u**3 - 30 * u**2 + 12 * u - 1),
    lambda u: 3 * (70 * u**4 - 140 * u**3 + 90 * u**2 - 20 * u + 1),
]


def neyman_uniformity_test(u):
    """Neyman's smooth test (order 4) of U(0, 1) for a vector of probabilities."""
    n = u.size
    statistic = sum((poly(u).sum() / np.sqrt(n)) ** 2 for poly in LEGENDRE)
    return float(statistic), float(sps.chi2.sf(statistic, len(LEGENDRE)))


def hawkins_tail_probabilities(X, groups):
    """Hawkins (1981) F statistics and their tail probabilities A_ij per group."""
    n, p = X.shape
    labels = np.unique(groups)
    g = labels.size
    nu = n - g
    centered = np.empty_like(X)
    for label in labels:
        rows = groups == label
        centered[rows] = X[rows] - X[rows].mean(axis=0)
    pooled = centered.T @ centered / nu
    inverse = np.linalg.inv(pooled)
    d2 = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    f_stat = (nu - p) * sizes * d2 / (p * ((sizes - 1) * nu - sizes * d2))
    return sps.f.sf(f_stat, p, nu - p)


def jamshidian_jalal_test(X, seed=config.RANDOM_SEED):
    keys, inverse = _patterns(X)
    if len(keys) < 2:
        return {"applicable": False}
    sizes = np.bincount(inverse)
    keep_groups = np.where(sizes >= MIN_GROUP_SIZE)[0]
    if keep_groups.size < 2:
        return {"applicable": False}
    rng = np.random.default_rng(seed)
    mu, sigma, _ = em_mvn(X)
    imputed = normal_imputation(X, mu, sigma, rng)
    rows = np.isin(inverse, keep_groups)
    groups = inverse[rows]
    tail = hawkins_tail_probabilities(imputed[rows], groups)

    group_p_values = []
    for label in np.unique(groups):
        _, p_value = neyman_uniformity_test(tail[groups == label])
        group_p_values.append(max(p_value, 1e-300))
    fisher = -2 * np.sum(np.log(group_p_values))
    hawkins_p = float(sps.chi2.sf(fisher, 2 * len(group_p_values)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad = sps.anderson_ksamp([tail[groups == label] for label in np.unique(groups)])
    nonparametric_p = float(ad.pvalue)

    alpha = config.SIGNIFICANCE
    if hawkins_p > alpha:
        decision = "Não rejeita MCAR (nem normalidade com homocedasticidade)."
        reject = False
    elif nonparametric_p <= alpha:
        decision = "Rejeita MCAR: as covariâncias diferem entre os padrões de ausência."
        reject = True
    else:
        decision = (
            "Hawkins rejeitou, mas o teste não paramétrico não: o motivo provável é "
            "falta de normalidade, não violação de MCAR."
        )
        reject = False
    return {
        "applicable": True,
        "hawkins_statistic": float(fisher),
        "hawkins_p": hawkins_p,
        "ad_statistic": float(ad.statistic),
        "nonparametric_p": nonparametric_p,
        "groups": int(len(group_p_values)),
        "decision": decision,
        "reject": reject,
    }


# ----------------------------------------------------------------------------
# Listing & Schlittgen nonparametric test
# ----------------------------------------------------------------------------

def listing_schlittgen_test(X, names):
    comparisons = []
    for k in range(X.shape[1]):
        missing = np.isnan(X[:, k])
        if not missing.any() or missing.all():
            continue
        for j in range(X.shape[1]):
            if j == k:
                continue
            values = X[:, j]
            group_missing = values[missing & ~np.isnan(values)]
            group_observed = values[~missing & ~np.isnan(values)]
            if group_missing.size < 2 or group_observed.size < 2:
                continue
            test = sps.mannwhitneyu(group_missing, group_observed, alternative="two-sided")
            comparisons.append(
                {
                    "missing_in": names[k],
                    "compared": names[j],
                    "median_missing": float(np.median(group_missing)),
                    "median_observed": float(np.median(group_observed)),
                    "statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            )
    if not comparisons:
        return {"applicable": False}
    m = len(comparisons)
    for item in comparisons:
        item["p_adjusted"] = min(1.0, item["p_value"] * m)  # Bonferroni
    global_p = min(item["p_adjusted"] for item in comparisons)
    return {
        "applicable": True,
        "comparisons": comparisons,
        "global_p": global_p,
        "reject": global_p <= config.SIGNIFICANCE,
    }


# ----------------------------------------------------------------------------
# Fairclough's logistic regression
# ----------------------------------------------------------------------------

def logistic_regression(X, y, max_iter=100):
    """Logistic regression via IRLS with Wald and likelihood-ratio tests."""
    design = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(design.shape[1])
    for _ in range(max_iter):
        prob = 1 / (1 + np.exp(-design @ beta))
        weights = prob * (1 - prob)
        hessian = design.T @ (design * weights[:, None])
        step = np.linalg.solve(hessian, design.T @ (y - prob))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    prob = np.clip(1 / (1 + np.exp(-design @ beta)), 1e-12, 1 - 1e-12)
    hessian = design.T @ (design * (prob * (1 - prob))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    loglik = float(np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob)))
    p0 = y.mean()
    loglik0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    lr = 2 * (loglik - loglik0)
    z = beta / se
    return {
        "coefficients": beta,
        "se": se,
        "z": z,
        "p_values": 2 * sps.norm.sf(np.abs(z)),
        "lr_statistic": lr,
        "lr_df": X.shape[1],
        "lr_p": float(sps.chi2.sf(lr, X.shape[1])),
    }


def fairclough_logistic_test(X, names):
    models = []
    for k in range(X.shape[1]):
        indicator = np.isnan(X[:, k]).astype(float)
        if indicator.sum() == 0 or indicator.sum() == len(indicator):
            continue
        predictors = [j for j in range(X.shape[1]) if j != k]
        Z = X[:, predictors]
        complete = ~np.isnan(Z).any(axis=1)
        Z = Z[complete]
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)  # standardized coefficients
        fit = logistic_regression(Z, indicator[complete])
        terms = [
            {
                "name": names[j],
                "coefficient": float(fit["coefficients"][i + 1]),
                "odds_ratio": float(np.exp(fit["coefficients"][i + 1])),
                "p_value": float(fit["p_values"][i + 1]),
            }
            for i, j in enumerate(predictors)
        ]
        models.append(
            {
                "missing_in": names[k],
                "terms": terms,
                "lr_statistic": float(fit["lr_statistic"]),
                "lr_df": int(fit["lr_df"]),
                "lr_p": fit["lr_p"],
                "reject": fit["lr_p"] <= config.SIGNIFICANCE,
            }
        )
    if not models:
        return {"applicable": False}
    return {"applicable": True, "models": models}


# ----------------------------------------------------------------------------
# Simulation lab: inject missing values with a known mechanism
# ----------------------------------------------------------------------------

def inject_missing(values, target_index, driver_index, mechanism, rate, seed):
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    n_missing = int(round(rate * n))
    if mechanism == "MCAR":
        weights = np.ones(n)
    else:
        source = values[:, driver_index] if mechanism == "MAR" else values[:, target_index]
        z = (source - source.mean()) / source.std(ddof=1)
        weights = np.exp(DEPENDENCE_STRENGTH * z)
    chosen = rng.choice(n, size=n_missing, replace=False, p=weights / weights.sum())
    mask = np.zeros(n, dtype=bool)
    mask[chosen] = True
    X = values.copy()
    X[mask, target_index] = np.nan
    return X, mask


@lru_cache(maxsize=64)
def run_simulation(mechanism, target, driver, rate, seed):
    df = get_dataset()
    names = config.VARIABLE_KEYS
    values = df[names].to_numpy(dtype=float)
    t, d = names.index(target), names.index(driver)
    X, mask = inject_missing(values, t, d, mechanism, rate, seed)
    return {
        "mask": mask,
        "true_target": values[:, t],
        "driver_values": values[:, d],
        "n_missing": int(mask.sum()),
        "n": int(mask.size),
        "little": little_mcar_test(X),
        "jamshidian_jalal": jamshidian_jalal_test(X, seed),
        "listing_schlittgen": listing_schlittgen_test(X, names),
        "fairclough": fairclough_logistic_test(X, names),
        "true_mean": float(values[:, t].mean()),
        "observed_mean": float(np.nanmean(X[:, t])),
        "missing_mean": float(values[mask, t].mean()),
    }


def real_data_tests():
    """Run the tests on the original base (no missing values -> not applicable)."""
    df = get_dataset()
    names = config.VARIABLE_KEYS
    X = df[names].to_numpy(dtype=float)
    return {
        "little": little_mcar_test(X),
        "jamshidian_jalal": jamshidian_jalal_test(X),
        "listing_schlittgen": listing_schlittgen_test(X, names),
        "fairclough": fairclough_logistic_test(X, names),
    }
