"""HTTP routes. Each page corresponds to a section of the assignment."""

from functools import lru_cache

import plotly
from flask import Blueprint, Response, render_template, request

from . import charts, config, data, missing, narrative, outliers, quality, stats
from .dictionary import COLUMNS, DERIVED_COLUMNS, DROP, KEEP, TARGET, label_for
from .formatting import date as format_date

bp = Blueprint("main", __name__)

NAV = [
    ("Introdução", [("main.index", "0", "Visão geral")]),
    (
        "Base de dados",
        [
            ("main.dataset", "1", "Estrutura e tratamento"),
            ("main.dictionary", "2", "Dicionário de dados"),
        ],
    ),
    (
        "Qualidade dos dados",
        [
            ("main.special", "3.1", "Valores especiais e consistência"),
            ("main.missing_values", "3.2", "Valores ausentes"),
            ("main.outliers_page", "3.3", "Outliers e inconsistências"),
        ],
    ),
    (
        "Estatística descritiva",
        [
            ("main.descriptive", "4.1", "Medidas, boxplot e barras"),
            ("main.skewness", "4.2", "Assimetria"),
            ("main.cv", "4.3", "Coeficiente de variação"),
            ("main.iqr", "4.4", "Intervalo interquartil"),
        ],
    ),
    ("Conclusões", [("main.synthesis", "5", "Síntese e interpretação")]),
]


@bp.app_context_processor
def inject_nav():
    return {"NAV": NAV}


def _choice(name, options, default):
    value = request.args.get(name, default)
    return value if value in options else default


def _team_choice(default="both"):
    return _choice("team", ["both"] + config.TEAM_CODES, default)


def _selected_teams(team):
    return config.TEAM_CODES if team == "both" else [team]


# ----------------------------------------------------------------------------
# Static asset: plotly.js bundled with the Python package (works offline)
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _plotly_js():
    return plotly.offline.get_plotlyjs()


@bp.route("/vendor/plotly.min.js")
def plotly_js():
    response = Response(_plotly_js(), mimetype="application/javascript")
    response.headers["Cache-Control"] = "public, max-age=604800"
    return response


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------

def _overview_numbers():
    raw = data.load_raw()
    df = data.get_dataset()
    log = data.get_treatment_log()
    per_team = quality.games_per_team()
    return {
        "rows": raw.shape[0],
        "columns": raw.shape[1],
        "games": df["game_id"].nunique(),
        "teams": per_team["teams"],
        "games_per_team_min": per_team["min"],
        "games_per_team_max": per_team["max"],
        "date_min": format_date(log["date_min"]),
        "date_max": format_date(log["date_max"]),
        "team_games": {team: int((df["team"] == team).sum()) for team in config.TEAM_CODES},
    }


@bp.route("/")
def index():
    summary = stats.team_variable_summary()
    return render_template(
        "index.html",
        page="main.index",
        overview=_overview_numbers(),
        summary=summary,
    )


@bp.route("/base")
def dataset():
    raw = data.load_raw()
    df = data.get_dataset()
    log = data.get_treatment_log()
    team = _choice("team", config.TEAM_CODES + [config.LEAGUE_CODE], "NYK")
    first = raw.iloc[0]
    type_rows = []
    for column in COLUMNS:
        original = column["original"]
        type_rows.append(
            {
                **column,
                "dtype": str(raw[original].dtype),
                "dtype_after": str(df[column["new"]].dtype),
                "example": first[original],
                "unique": int(raw[original].nunique()),
            }
        )
    derived_preview = data.team_data(team).sort_values("game_date").head(8)[
        ["game_date", "matchup", "team", "opponent", "is_home", "win", "game_month", "game_weekday",
         "game_number", "days_since_previous_game", "is_back_to_back", "overtime_periods"]
    ]
    ml = data.ml_ready(df)
    semantic_counts = data.semantic_type_counts()
    return render_template(
        "dataset.html",
        page="main.dataset",
        overview=_overview_numbers(),
        log=log,
        type_rows=type_rows,
        semantic_counts=semantic_counts,
        file_size=data.file_size_bytes(),
        memory_raw=data.memory_usage_bytes(raw),
        memory_treated=data.memory_usage_bytes(df),
        derived_preview=derived_preview,
        team=team,
        columns=COLUMNS,
        derived=DERIVED_COLUMNS,
        ml_shape=ml.shape,
        ml_columns=list(ml.columns),
        dropped=[c for c in COLUMNS if c["ml"] == DROP],
        kept=[c for c in COLUMNS + DERIVED_COLUMNS if c["ml"] == KEEP],
        target=[c for c in COLUMNS if c["ml"] == TARGET],
        month_chart=charts.to_json(charts.games_per_month(df)),
        home_counts=df["is_home"].value_counts().to_dict(),
        b2b=int(df["is_back_to_back"].sum()),
        overtime=df["overtime_periods"].value_counts().sort_index().to_dict(),
    )


@bp.route("/dicionario")
def dictionary():
    raw = data.load_raw()
    return render_template(
        "dictionary.html",
        page="main.dictionary",
        columns=COLUMNS,
        derived=DERIVED_COLUMNS,
        examples=raw.iloc[0].to_dict(),
    )


@bp.route("/qualidade/especiais")
def special():
    rows, duplicates = quality.special_values()
    rules = quality.consistency_rules()
    return render_template(
        "special.html",
        page="main.special",
        rows=rows,
        duplicates=duplicates,
        rules=rules,
        n_rules_ok=sum(rule["ok"] for rule in rules),
        overview=_overview_numbers(),
    )


@bp.route("/qualidade/ausentes")
def missing_values():
    rows, patterns = missing.missing_summary()
    mechanism = _choice("mechanism", list(missing.MECHANISMS), "MAR")
    target = _choice("target", config.VARIABLE_KEYS, "points")
    driver_options = [v for v in config.VARIABLE_KEYS if v != target]
    driver = _choice("driver", driver_options, "assists" if target != "assists" else "points")
    rate_text = request.args.get("rate", "0.2")
    rate = float(rate_text) if rate_text in {str(r) for r in missing.RATES} else 0.2
    try:
        seed = max(0, min(int(request.args.get("seed", config.RANDOM_SEED)), 10_000))
    except ValueError:
        seed = config.RANDOM_SEED
    sim = missing.run_simulation(mechanism, target, driver, rate, seed)
    alpha = config.SIGNIFICANCE
    verdicts = {
        "Little": sim["little"]["p_value"] <= alpha,
        "Jamshidian-Jalal": sim["jamshidian_jalal"]["reject"],
        "Listing-Schlittgen": sim["listing_schlittgen"]["reject"],
        "Regressão logística de Fairclough": sim["fairclough"]["models"][0]["reject"],
    }
    logistic = sim["fairclough"]["models"][0]
    significant_predictors = [term["name"] for term in logistic["terms"] if term["p_value"] <= alpha]
    return render_template(
        "missing.html",
        page="main.missing_values",
        rows=rows,
        patterns=patterns,
        total_missing=sum(r["missing"] for r in rows),
        real=missing.real_data_tests(),
        mechanisms=missing.MECHANISMS,
        rates=missing.RATES,
        mechanism=mechanism,
        target=target,
        driver=driver,
        driver_options=driver_options,
        rate=rate,
        seed=seed,
        sim=sim,
        sim_chart=charts.to_json(charts.missing_simulation_chart(sim, target, driver, mechanism)),
        alpha=alpha,
        rejected=[name for name, value in verdicts.items() if value],
        not_rejected=[name for name, value in verdicts.items() if not value],
        significant_predictors=significant_predictors,
        logistic_rejects=logistic["reject"],
        label_for=label_for,
    )


@bp.route("/qualidade/outliers")
def outliers_page():
    scope = _choice("scope", [config.LEAGUE_CODE] + config.TEAM_CODES, config.LEAGUE_CODE)
    k = 3.0 if request.args.get("k") == "3" else config.IQR_K
    df = data.team_data(scope)
    columns = data.numeric_quality_columns(df)
    column = _choice("column", columns, "points")
    table = outliers.univariate_table(scope, k)
    detail, info = outliers.univariate_detail(scope, column, k)
    flagged = detail[detail["n_methods"] > 0].sort_values(["n_methods", column], ascending=[False, False])
    multi = outliers.multivariate_analysis(scope, k)
    games = multi["games"]
    flagged_multi = games[games["n_methods"] > 0].sort_values(
        ["n_methods", "lof_score"], ascending=[False, False]
    )
    strong = games[games["n_methods"] >= 3]
    meta = next((c for c in COLUMNS if c["new"] == column), {"label": column, "unit": ""})
    totals = {key: sum(row[key] for row in table) for key in outliers.UNIVARIATE_METHODS}
    return render_template(
        "outliers.html",
        page="main.outliers_page",
        scope=scope,
        scope_options={config.LEAGUE_CODE: config.LEAGUE_LABEL, **{t: config.TEAMS[t]["name"] for t in config.TEAM_CODES}},
        k=k,
        columns=columns,
        column=column,
        column_label=meta["label"],
        table=table,
        totals=totals,
        univariate_methods=outliers.UNIVARIATE_METHODS,
        multivariate_methods=outliers.MULTIVARIATE_METHODS,
        flagged=flagged.head(40),
        n_flagged=len(flagged),
        info=info,
        strip_chart=charts.to_json(charts.univariate_outlier_strip(detail, column, meta["label"], meta["unit"])),
        multi=multi,
        flagged_multi=flagged_multi.head(30),
        n_flagged_multi=len(flagged_multi),
        pca_chart=charts.to_json(charts.pca_scatter(games, multi["explained_variance"])),
        kdist_chart=charts.to_json(
            charts.k_distance_plot(
                multi["results"]["dbscan"]["k_distances"],
                multi["results"]["dbscan"]["eps"],
                multi["results"]["dbscan"]["min_samples"],
            )
        ),
        label_for=label_for,
        n=len(df),
        strong_games=len(strong),
        strong_with_driver=int(strong["drivers"].map(bool).sum()),
    )


@bp.route("/descritiva")
def descriptive():
    variable = _choice("variable", config.VARIABLE_KEYS, "points")
    team = _team_choice()
    scale = _choice("scale", ["raw", "z"], "raw")
    summary = stats.team_variable_summary()
    reference = stats.league_reference()
    zscores = stats.team_mean_zscores()
    teams = _selected_teams(team)
    box = charts.boxplot(variable, {t: summary[variable][t]["describe"] for t in teams}, teams)
    bars = charts.bar_zscores(zscores, reference) if scale == "z" else charts.bar_means_small_multiples(summary, reference)
    outlier_lists = {t: stats.outlier_games(data.team_data(t), variable) for t in teams}
    return render_template(
        "descriptive.html",
        page="main.descriptive",
        variable=variable,
        team=team,
        teams=teams,
        scale=scale,
        summary=summary,
        reference=reference,
        zscores=zscores,
        box_chart=charts.to_json(box),
        bar_chart=charts.to_json(bars),
        outlier_lists=outlier_lists,
        texts=narrative.descriptive_variable(variable, summary, reference),
        z_texts=narrative.zscore_explanation(zscores),
    )


@bp.route("/assimetria")
def skewness():
    variable = _choice("variable", config.VARIABLE_KEYS, "rebounds")
    team = _team_choice()
    teams = _selected_teams(team)
    summary = stats.team_variable_summary()
    histograms = {
        t: charts.to_json(
            charts.histogram_kde(
                variable, t, data.team_data(t)[variable], summary[variable][t]["describe"], summary[variable][t]["skew"]
            )
        )
        for t in teams
    }
    return render_template(
        "skewness.html",
        page="main.skewness",
        variable=variable,
        team=team,
        teams=teams,
        summary=summary,
        histograms=histograms,
        bars_chart=charts.to_json(charts.skewness_bars(summary)),
        variable_texts=narrative.skewness_variable(variable, summary),
        overview_texts=narrative.skewness_overview(summary),
    )


@bp.route("/cv")
def cv():
    variable = _choice("variable", config.VARIABLE_KEYS, "blocks")
    summary = stats.team_variable_summary()
    return render_template(
        "cv.html",
        page="main.cv",
        variable=variable,
        summary=summary,
        chart=charts.to_json(charts.cv_bars(summary)),
        variable_texts=narrative.cv_variable(variable, summary),
        overview_texts=narrative.cv_overview(summary),
        classify=stats.classify_cv,
    )


@bp.route("/iqr")
def iqr():
    variable = _choice("variable", config.VARIABLE_KEYS, "assists")
    summary = stats.team_variable_summary()
    return render_template(
        "iqr.html",
        page="main.iqr",
        variable=variable,
        summary=summary,
        dispersion_chart=charts.to_json(charts.dispersion_comparison(variable, summary)),
        ratio_chart=charts.to_json(charts.iqr_ratio_bars(summary)),
        variable_texts=narrative.iqr_variable(variable, summary),
        overview_texts=narrative.iqr_overview(summary),
    )


@bp.route("/sintese")
def synthesis():
    summary = stats.team_variable_summary()
    reference = stats.league_reference()
    zscores = stats.team_mean_zscores()
    overview = _overview_numbers()
    rules = quality.consistency_rules()
    quality_summary = {**overview, "rules": len(rules)}
    outlier_summary = {
        "tukey_total": sum(
            summary[v][t]["describe"]["n_outliers"] for v in config.VARIABLE_KEYS for t in config.TEAM_CODES
        )
    }
    return render_template(
        "synthesis.html",
        page="main.synthesis",
        summary=summary,
        reference=reference,
        zscores=zscores,
        texts=narrative.synthesis(summary, reference, zscores, quality_summary, outlier_summary),
        z_chart=charts.to_json(charts.bar_zscores(zscores, reference)),
    )
