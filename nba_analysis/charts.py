"""Plotly figures. Every chart has a title, labelled axes with units and a legend."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sps

from . import config
from .formatting import number

INK = "#1f1f1d"
MUTED = "#6b6a66"
GRID = "#e6e5e0"
OUTLIER_COLOR = "#c4314b"
FONT_FAMILY = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def to_json(fig):
    return fig.to_json()


def _layout(fig, title, x_title=None, y_title=None, height=440, subtitle=None, legend_rows=1, has_x_title=True):
    """Common styling. The legend sits below the x-axis title, sized by its number of rows."""
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
    top = 86 if subtitle else 64
    gap = 66 if (has_x_title or x_title) else 34
    bottom = gap + 22 * legend_rows + 14
    plot_height = max(height - top - bottom, 120)
    fig.update_layout(
        title={"text": text, "x": 0, "xanchor": "left", "y": 0.97, "yanchor": "top", "font": {"size": 16}},
        template="plotly_white",
        font={"family": FONT_FAMILY, "size": 13, "color": INK},
        separators=",.",
        height=height,
        margin={"l": 70, "r": 24, "t": top, "b": bottom},
        hoverlabel={"font": {"size": 12, "family": FONT_FAMILY}},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=True,
    )
    fig.update_layout(
        legend={
            "orientation": "h", "yanchor": "top", "y": -gap / plot_height,
            "xanchor": "left", "x": 0, "font": {"size": 12},
        }
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c9c8c2")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c9c8c2")
    return fig


def _rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _var_title(variable):
    meta = config.ANALYSIS_VARIABLES[variable]
    return f"{meta['label']} ({meta['unit']})"


def _spread_labels(items, min_gap):
    """Shift label positions so that consecutive labels are at least `min_gap` apart."""
    items = sorted(items, key=lambda item: item["y"])
    positions = [item["y"] for item in items]
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap
    # Re-center the block so labels stay close to their true values.
    shift = (np.mean([item["y"] for item in items]) - np.mean(positions)) if items else 0
    for item, position in zip(items, positions):
        item["label_y"] = position + shift
    return items


# ----------------------------------------------------------------------------
# Boxplot with every element required by the assignment
# ----------------------------------------------------------------------------

def boxplot(variable, summaries, teams):
    """summaries[team] -> output of stats.describe."""
    fig = go.Figure()
    meta = config.ANALYSIS_VARIABLES[variable]
    positions = {team: i for i, team in enumerate(teams)}
    all_values = []
    for team in teams:
        d = summaries[team]
        all_values += [d["min"], d["max"], d["lower_fence"], d["upper_fence"]]
    y_span = max(all_values) - min(all_values) or 1

    lines = {"upper": ([], []), "lower": ([], [])}
    means = ([], [], [])
    mode_points = ([], [], [])
    out_high = ([], [], [])
    out_low = ([], [], [])
    rng = np.random.default_rng(config.RANDOM_SEED)
    annotations = []

    for team in teams:
        d = summaries[team]
        x = positions[team]
        color = config.TEAMS[team]["color"]
        fig.add_trace(
            go.Box(
                x=[x],
                q1=[d["q1"]],
                median=[d["median"]],
                q3=[d["q3"]],
                lowerfence=[d["whisker_low"]],
                upperfence=[d["whisker_high"]],
                name=f"{config.TEAMS[team]['name']} (caixa Q1–Q3, mediana e bigodes)",
                marker_color=color,
                fillcolor=_rgba(color, 0.18),
                line={"color": color, "width": 2},
                width=0.38,
                boxpoints=False,
                hoverinfo="y",
            )
        )
        for key, value in (("upper", d["upper_fence"]), ("lower", d["lower_fence"])):
            lines[key][0].extend([x - 0.3, x + 0.3, None])
            lines[key][1].extend([value, value, None])
        means[0].append(x)
        means[1].append(d["mean"])
        means[2].append(config.TEAMS[team]["name"])
        for mode_value in d["modes"]:
            mode_points[0].append(x)
            mode_points[1].append(mode_value)
            mode_points[2].append(config.TEAMS[team]["name"])
        for value in d["outliers_high"]:
            out_high[0].append(x + rng.uniform(-0.06, 0.06))
            out_high[1].append(value)
            out_high[2].append(config.TEAMS[team]["name"])
        for value in d["outliers_low"]:
            out_low[0].append(x + rng.uniform(-0.06, 0.06))
            out_low[1].append(value)
            out_low[2].append(config.TEAMS[team]["name"])

        mode_text = " / ".join(number(m) for m in d["modes"][:3]) + (" …" if len(d["modes"]) > 3 else "")
        labels = [
            {"y": d["upper_fence"], "text": f"LS = {number(d['upper_fence'])}"},
            {"y": d["q3"], "text": f"Q3 = {number(d['q3'])}"},
            {"y": d["median"], "text": f"Mediana = {number(d['median'])}"},
            {"y": d["mean"], "text": f"Média = {number(d['mean'])}"},
            {"y": d["modes"][0], "text": f"Moda = {mode_text}"},
            {"y": d["q1"], "text": f"Q1 = {number(d['q1'])}"},
            {"y": d["lower_fence"], "text": f"LI = {number(d['lower_fence'])}"},
        ]
        for item in _spread_labels(labels, 0.055 * y_span):
            annotations.append(
                {
                    "x": x + 0.2,
                    "y": item["y"],
                    "ax": x + 0.36,
                    "ay": item["label_y"],
                    "axref": "x",
                    "ayref": "y",
                    "xref": "x",
                    "yref": "y",
                    "text": item["text"],
                    "showarrow": True,
                    "arrowhead": 0,
                    "arrowwidth": 1,
                    "arrowcolor": "#b5b4ad",
                    "xanchor": "left",
                    "font": {"size": 11, "color": MUTED},
                }
            )

    fig.add_trace(
        go.Scatter(
            x=lines["upper"][0], y=lines["upper"][1], mode="lines",
            name=f"Limite superior (Q3 + {number(config.IQR_K, 1)}·IQR)",
            line={"color": "#7a7974", "dash": "dash", "width": 1.5},
            hovertemplate="Limite superior: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lines["lower"][0], y=lines["lower"][1], mode="lines",
            name=f"Limite inferior (Q1 − {number(config.IQR_K, 1)}·IQR)",
            line={"color": "#7a7974", "dash": "dot", "width": 1.5},
            hovertemplate="Limite inferior: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=means[0], y=means[1], text=means[2], mode="markers", name="Média",
            marker={"symbol": "diamond", "size": 11, "color": INK, "line": {"color": "#ffffff", "width": 2}},
            hovertemplate="%{text}<br>Média: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mode_points[0], y=mode_points[1], text=mode_points[2], mode="markers", name="Moda",
            marker={"symbol": "star", "size": 13, "color": "#e9b200", "line": {"color": INK, "width": 1}},
            hovertemplate="%{text}<br>Moda: %{y:.2f}<extra></extra>",
        )
    )
    for points in (out_high, out_low):
        if not points[0]:
            points[0].append(None)
            points[1].append(None)
            points[2].append("")
    fig.add_trace(
        go.Scatter(
            x=out_high[0], y=out_high[1], text=out_high[2], mode="markers",
            name="Outliers superiores" + ("" if out_high[1][0] is not None else " (nenhum)"),
            marker={"symbol": "triangle-up", "size": 10, "color": OUTLIER_COLOR, "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="%{text}<br>Outlier superior: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=out_low[0], y=out_low[1], text=out_low[2], mode="markers",
            name="Outliers inferiores" + ("" if out_low[1][0] is not None else " (nenhum)"),
            marker={"symbol": "triangle-down", "size": 10, "color": OUTLIER_COLOR, "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="%{text}<br>Outlier inferior: %{y:.2f}<extra></extra>",
        )
    )
    _layout(
        fig,
        f"Boxplot — {meta['label']} por jogo",
        x_title="Time",
        y_title=_var_title(variable),
        height=600,
        subtitle="Quartis por interpolação linear (tipo 7); limites de Tukey com k = 1,5",
        legend_rows=4,
    )
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(
        tickvals=list(positions.values()),
        ticktext=[config.TEAMS[t]["name"] for t in teams],
        range=[-0.55, len(teams) - 1 + 0.95],
        showgrid=False,
    )
    return fig


# ----------------------------------------------------------------------------
# Bar charts comparing the teams
# ----------------------------------------------------------------------------

def bar_means_small_multiples(summary, reference):
    rows, cols = 2, 4
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[
            f"{config.ANALYSIS_VARIABLES[v]['label']} ({config.ANALYSIS_VARIABLES[v]['abbr']})"
            for v in config.VARIABLE_KEYS
        ],
        horizontal_spacing=0.07, vertical_spacing=0.2,
    )
    for i, variable in enumerate(config.VARIABLE_KEYS):
        row, col = i // cols + 1, i % cols + 1
        for team in config.TEAM_CODES:
            d = summary[variable][team]["describe"]
            fig.add_trace(
                go.Bar(
                    x=[config.TEAMS[team]["short"]],
                    y=[d["mean"]],
                    error_y={"type": "data", "array": [d["std"]], "color": "#55544f", "thickness": 1.2, "width": 4},
                    marker={"color": config.TEAMS[team]["color"], "cornerradius": 4},
                    name=config.TEAMS[team]["name"],
                    legendgroup=team,
                    showlegend=i == 0,
                    hovertemplate=f"{config.TEAMS[team]['name']}<br>Média: %{{y:.2f}}<br>"
                    f"Desvio padrão: {number(d['std'])}<extra></extra>",
                ),
                row=row, col=col,
            )
        league_mean = reference[variable]["game_mean"]
        fig.add_trace(
            go.Scatter(
                x=[config.TEAMS[t]["short"] for t in config.TEAM_CODES], y=[league_mean] * 2,
                mode="lines", line={"color": INK, "dash": "dash", "width": 1.5},
                name="Média da liga", legendgroup="league", showlegend=i == 0,
                hovertemplate=f"Média da liga: {number(league_mean)}<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.update_yaxes(title_text=config.ANALYSIS_VARIABLES[variable]["unit_short"] + "/jogo", row=row, col=col)
    _layout(
        fig,
        "Média por jogo de cada variável — Knicks × Suns",
        height=620,
        subtitle="Barras = média; linha de erro = ±1 desvio padrão amostral; tracejado = média da liga",
    )
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(title_text="Time", row=2)
    fig.update_xaxes(gridcolor=GRID)
    fig.add_annotation(
        x=0.99, y=0.2, xref="paper", yref="paper", showarrow=False, align="left",
        text="Cada painel tem sua própria<br>escala (unidades diferentes).<br>"
        "Para comparar variáveis entre si,<br>use a visão em z-score.",
        font={"size": 12, "color": MUTED}, xanchor="right",
    )
    return fig


def bar_zscores(zscores, reference):
    labels = [config.ANALYSIS_VARIABLES[v]["label"] for v in config.VARIABLE_KEYS]
    fig = go.Figure()
    for team in config.TEAM_CODES:
        values = [zscores[v][team] for v in config.VARIABLE_KEYS]
        means = [reference[v]["team_means"][team] for v in config.VARIABLE_KEYS]
        fig.add_trace(
            go.Bar(
                y=labels, x=values, orientation="h", name=config.TEAMS[team]["name"],
                marker={"color": config.TEAMS[team]["color"], "cornerradius": 4},
                customdata=np.array(means).reshape(-1, 1),
                hovertemplate="%{y}<br>z = %{x:.2f}<br>Média do time: %{customdata[0]:.2f}<extra>"
                + config.TEAMS[team]["short"] + "</extra>",
            )
        )
    for value, dash in ((-1, "dot"), (1, "dot")):
        fig.add_vline(x=value, line={"color": "#a9a8a2", "dash": dash, "width": 1})
    fig.add_vline(x=0, line={"color": INK, "width": 1.5})
    _layout(
        fig,
        "Médias padronizadas (z-score) — comparação com os 30 times da NBA",
        x_title="z-score da média do time (desvios-padrão em relação à média dos 30 times)",
        y_title="Variável",
        height=500,
        subtitle="z = (média do time − média das médias dos 30 times) / desvio padrão das médias dos 30 times",
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed")
    return fig


# ----------------------------------------------------------------------------
# Histogram + KDE + boxplot (skewness)
# ----------------------------------------------------------------------------

def _bin_size(x):
    q1, q3 = np.percentile(x, [25, 75])
    width = 2 * (q3 - q1) / np.cbrt(len(x)) if q3 > q1 else 1  # Freedman-Diaconis
    return max(1.0, float(np.round(width)))


def histogram_kde(variable, team, values, describe_result, skew_result):
    x = np.asarray(values, dtype=float)
    color = config.TEAMS[team]["color"]
    meta = config.ANALYSIS_VARIABLES[variable]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.04)
    size = _bin_size(x)
    start = np.floor(x.min()) - 0.5
    fig.add_trace(
        go.Histogram(
            x=x, histnorm="probability density",
            xbins={"start": start, "end": x.max() + size, "size": size},
            marker={"color": _rgba(color, 0.45), "line": {"color": "#ffffff", "width": 1}},
            name=f"Histograma (densidade, classes de {number(size, 0)})",
            hovertemplate="Classe: %{x}<br>Densidade: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    kde = sps.gaussian_kde(x)
    grid = np.linspace(x.min() - 2 * size, x.max() + 2 * size, 300)
    fig.add_trace(
        go.Scatter(
            x=grid, y=kde(grid), mode="lines", name="Curva de densidade (KDE gaussiano)",
            line={"color": color, "width": 2.5},
            hovertemplate="Valor: %{x:.1f}<br>Densidade: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    top = float(max(kde(grid).max(), np.histogram(x, bins=np.arange(start, x.max() + size + 1, size), density=True)[0].max()))
    for value, name, dash in (
        (describe_result["mean"], "Média", "solid"),
        (describe_result["median"], "Mediana", "dash"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[value, value], y=[0, top * 1.05], mode="lines", name=f"{name} = {number(value)}",
                line={"color": INK, "dash": dash, "width": 1.8},
                hovertemplate=f"{name}: {number(value)}<extra></extra>",
            ),
            row=1, col=1,
        )
    d = describe_result
    fig.add_trace(
        go.Box(
            y=[0], q1=[d["q1"]], median=[d["median"]], q3=[d["q3"]],
            lowerfence=[d["whisker_low"]], upperfence=[d["whisker_high"]],
            orientation="h", name="Boxplot", marker_color=color, fillcolor=_rgba(color, 0.18),
            line={"color": color}, boxpoints=False, showlegend=False, hoverinfo="x",
        ),
        row=2, col=1,
    )
    outliers = d["outliers_low"] + d["outliers_high"]
    if outliers:
        fig.add_trace(
            go.Scatter(
                x=outliers, y=[0] * len(outliers), mode="markers", name="Outliers (1,5·IQR)",
                marker={"color": OUTLIER_COLOR, "size": 9, "symbol": "circle-open", "line": {"width": 2}},
                hovertemplate="Outlier: %{x}<extra></extra>",
            ),
            row=2, col=1,
        )
    _layout(
        fig,
        f"{config.TEAMS[team]['short']} — {meta['label'].lower()}",
        height=560,
        subtitle=f"G1 = {number(skew_result['G1'])} · {skew_result['direction']}, {skew_result['intensity']}",
        legend_rows=5,
    )
    fig.update_xaxes(title_text=_var_title(variable), row=2, col=1)
    fig.update_yaxes(title_text="Densidade", row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text="Box", row=2, col=1)
    fig.update_layout(bargap=0.02)
    return fig


def skewness_bars(summary):
    labels = [config.ANALYSIS_VARIABLES[v]["label"] for v in config.VARIABLE_KEYS]
    fig = go.Figure()
    for team in config.TEAM_CODES:
        fig.add_trace(
            go.Bar(
                x=labels, y=[summary[v][team]["skew"]["G1"] for v in config.VARIABLE_KEYS],
                name=config.TEAMS[team]["name"], marker={"color": config.TEAMS[team]["color"], "cornerradius": 4},
                hovertemplate="%{x}<br>G1 = %{y:.2f}<extra>" + config.TEAMS[team]["short"] + "</extra>",
            )
        )
    for value in (-1, -0.5, 0.5, 1):
        fig.add_hline(
            y=value, line={"color": "#a9a8a2", "dash": "dot" if abs(value) == 0.5 else "dash", "width": 1}
        )
    fig.add_hline(y=0, line={"color": INK, "width": 1.2})
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line={"color": "#a9a8a2", "dash": "dot"}, name="|G1| = 0,5 (moderada)")
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", line={"color": "#a9a8a2", "dash": "dash"}, name="|G1| = 1 (forte)")
    )
    _layout(
        fig, "Coeficiente de assimetria (G1) por variável e time",
        x_title="Variável", y_title="G1 (adimensional)", height=480, legend_rows=2,
    )
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


# ----------------------------------------------------------------------------
# Coefficient of variation
# ----------------------------------------------------------------------------

def cv_bars(summary):
    labels = [config.ANALYSIS_VARIABLES[v]["label"] for v in config.VARIABLE_KEYS]
    fig = go.Figure()
    for team in config.TEAM_CODES:
        fig.add_trace(
            go.Bar(
                x=labels, y=[summary[v][team]["cv"]["cv"] for v in config.VARIABLE_KEYS],
                name=f"{config.TEAMS[team]['name']} — CV observado",
                marker={"color": config.TEAMS[team]["color"], "cornerradius": 4},
                offsetgroup=team,
                hovertemplate="%{x}<br>CV = %{y:.2f}%<extra>" + config.TEAMS[team]["short"] + "</extra>",
            )
        )
    for i, team in enumerate(config.TEAM_CODES):
        fig.add_trace(
            go.Scatter(
                x=labels, y=[summary[v][team]["cv"]["cv_poisson"] for v in config.VARIABLE_KEYS],
                mode="markers", name="CV esperado se a contagem fosse Poisson (100/√média)" if i == 0 else None,
                showlegend=i == 0, legendgroup="poisson",
                marker={"symbol": "line-ew", "size": 22, "line": {"color": INK, "width": 2.5}},
                offsetgroup=team,
                hovertemplate="%{x}<br>CV Poisson = %{y:.2f}%<extra>" + config.TEAMS[team]["short"] + "</extra>",
            )
        )
    _layout(
        fig, "Coeficiente de variação (CV) por variável e time",
        x_title="Variável", y_title="CV (%)", height=500,
        subtitle="CV = s / média × 100, com s amostral (n − 1)", legend_rows=2,
    )
    fig.update_layout(barmode="group", bargap=0.3, scattermode="group")
    return fig


# ----------------------------------------------------------------------------
# IQR vs range vs standard deviation
# ----------------------------------------------------------------------------

def dispersion_comparison(variable, summary):
    meta = config.ANALYSIS_VARIABLES[variable]
    fig = go.Figure()
    teams = config.TEAM_CODES
    names = [config.TEAMS[t]["name"] for t in teams]
    for i, team in enumerate(teams):
        d = summary[variable][team]["describe"]
        color = config.TEAMS[team]["color"]
        y = names[i]
        fig.add_trace(
            go.Scatter(
                x=[d["min"], d["max"]], y=[y, y], mode="lines+markers",
                line={"color": "#b5b4ad", "width": 2}, marker={"symbol": "line-ns", "size": 16, "line": {"width": 2, "color": "#7a7974"}},
                name="Amplitude (mín. → máx.)", legendgroup="range", showlegend=i == 0,
                hovertemplate=f"Mín.: {number(d['min'])} · Máx.: {number(d['max'])}<br>Amplitude: {number(d['range'])}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[d["mean"] - d["std"], d["mean"] + d["std"]], y=[y, y], mode="lines",
                line={"color": INK, "width": 2, "dash": "dot"},
                name="Média ± 1 desvio padrão", legendgroup="std", showlegend=i == 0,
                hovertemplate=f"Média ± s: {number(d['mean'] - d['std'])} a {number(d['mean'] + d['std'])}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=[d["iqr"]], base=[d["q1"]], y=[y], orientation="h", width=0.32,
                marker={"color": _rgba(color, 0.55), "line": {"color": color, "width": 2}},
                name=f"IQR {config.TEAMS[team]['short']} (Q1 → Q3)",
                hovertemplate=f"Q1: {number(d['q1'])} · Q3: {number(d['q3'])}<br>IQR: {number(d['iqr'])}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[d["median"]], y=[y], mode="markers", marker={"symbol": "line-ns", "size": 24, "line": {"width": 3, "color": INK}},
                name="Mediana", legendgroup="median", showlegend=i == 0,
                hovertemplate=f"Mediana: {number(d['median'])}<extra></extra>",
            )
        )
    _layout(
        fig, f"{meta['label']}: IQR × amplitude × desvio padrão",
        x_title=_var_title(variable), y_title="Time", height=440,
        subtitle="A faixa colorida contém os 50% centrais dos jogos", legend_rows=2,
    )
    fig.update_layout(barmode="overlay")
    return fig


def iqr_ratio_bars(summary):
    labels = [config.ANALYSIS_VARIABLES[v]["label"] for v in config.VARIABLE_KEYS]
    fig = go.Figure()
    for team in config.TEAM_CODES:
        fig.add_trace(
            go.Bar(
                x=labels, y=[summary[v][team]["iqr"]["range_over_iqr"] for v in config.VARIABLE_KEYS],
                name=config.TEAMS[team]["name"], marker={"color": config.TEAMS[team]["color"], "cornerradius": 4},
                hovertemplate="%{x}<br>Amplitude / IQR = %{y:.2f}<extra>" + config.TEAMS[team]["short"] + "</extra>",
            )
        )
    _layout(
        fig, "Quantas vezes a amplitude é maior que o IQR",
        x_title="Variável", y_title="Amplitude ÷ IQR (adimensional)", height=440,
        subtitle="Razões altas indicam caudas longas ou valores extremos isolados",
    )
    fig.update_layout(barmode="group", bargap=0.3)
    return fig


# ----------------------------------------------------------------------------
# Data-quality charts
# ----------------------------------------------------------------------------

def games_per_month(df):
    counts = df.groupby(df["game_date"].dt.to_period("M")).size()
    from .formatting import MONTHS_PT

    labels = [f"{MONTHS_PT[p.month]}/{str(p.year)[2:]}" for p in counts.index]
    fig = go.Figure(
        go.Bar(
            x=labels, y=counts.to_numpy(), name="Registros (time × jogo)",
            marker={"color": "#2a78d6", "cornerradius": 4},
            hovertemplate="%{x}: %{y} registros<extra></extra>",
        )
    )
    _layout(
        fig, "Registros por mês após a conversão de GAME_DATE para data",
        x_title="Mês da temporada 2025-26", y_title="Nº de registros (linhas)", height=380,
    )
    return fig


def univariate_outlier_strip(detail, column, label, unit):
    fig = go.Figure()
    rng = np.random.default_rng(config.RANDOM_SEED)
    jitter = rng.uniform(-0.35, 0.35, len(detail))
    palette = ["#c9c8c2", "#f2b48f", "#e8794a", "#d4472a", "#a3221c", "#6e0f0f"]
    for count in sorted(detail["n_methods"].unique()):
        part = detail[detail["n_methods"] == count]
        idx = detail.index.get_indexer(part.index)
        fig.add_trace(
            go.Scatter(
                x=part[column], y=jitter[idx], mode="markers",
                name=(f"Sinalizado por {count} {'método' if count == 1 else 'métodos'}" if count else "Não sinalizado"),
                marker={"color": palette[min(count, 5)], "size": 8 if count else 6, "line": {"color": "#ffffff", "width": 0.5}},
                customdata=np.column_stack([part["team"], part["matchup"], part["game_date"].dt.strftime("%d/%m/%Y")]),
                hovertemplate="%{customdata[1]} — %{customdata[2]}<br>Valor: %{x}<extra></extra>",
            )
        )
    _layout(
        fig, f"{label}: valores por jogo e consenso entre métodos univariados",
        x_title=f"Valor no jogo ({unit})", y_title="Jogos", height=420, legend_rows=2,
    )
    fig.update_yaxes(showticklabels=False, range=[-0.6, 0.6], showgrid=False)
    return fig


def pca_scatter(games, explained):
    fig = go.Figure()
    palette = ["#c9c8c2", "#f2b48f", "#e8794a", "#c4314b", "#6e0f0f"]
    for count in range(5):
        part = games[games["n_methods"] == count]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["pc1"], y=part["pc2"], mode="markers",
                name=f"Sinalizado por {count} de 4 métodos" if count else "Não sinalizado",
                marker={"color": palette[count], "size": 7 if count == 0 else 10, "line": {"color": "#ffffff", "width": 0.8}},
                customdata=np.column_stack([part["matchup"], part["game_date"].dt.strftime("%d/%m/%Y")]),
                hovertemplate="%{customdata[0]} — %{customdata[1]}<extra></extra>",
            )
        )
    _layout(
        fig, "Jogos nos 2 primeiros componentes principais",
        x_title=f"Componente principal 1 ({number(explained[0] * 100)}% da variância)",
        y_title=f"Componente principal 2 ({number(explained[1] * 100)}% da variância)",
        height=600,
        subtitle="Cor = nº de métodos que sinalizaram o jogo",
        legend_rows=5,
    )
    return fig


def k_distance_plot(k_distances, eps, min_samples):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(k_distances) + 1), y=k_distances, mode="lines",
            name=f"Distância ao {min_samples}º vizinho (ordenada)", line={"color": "#2a78d6", "width": 2},
            hovertemplate="Posição %{x}<br>Distância: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1, len(k_distances)], y=[eps, eps], mode="lines",
            name=f"eps escolhido no joelho = {number(eps)}", line={"color": OUTLIER_COLOR, "dash": "dash", "width": 1.5},
        )
    )
    _layout(
        fig, "Gráfico k-distância (escolha do eps)",
        x_title="Jogos ordenados pela distância", y_title="Distância euclidiana (em z)",
        height=600, legend_rows=2, subtitle="DBSCAN: eps no joelho da curva",
    )
    return fig


def missing_simulation_chart(sim, target, driver, mechanism):
    mask = sim["mask"]
    target_meta = config.ANALYSIS_VARIABLES[target]
    driver_meta = config.ANALYSIS_VARIABLES[driver]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=[
            f"{driver_meta['label']} (variável observada)",
            f"{target_meta['label']} (valor real, conhecido só na simulação)",
        ],
    )
    for col, values in ((1, sim["driver_values"]), (2, sim["true_target"])):
        for missing, name, color in ((False, "Linhas com valor observado", "#8a8984"), (True, "Linhas que ficaram ausentes", OUTLIER_COLOR)):
            short = "Ausentes" if missing else "Observadas"
            fig.add_trace(
                go.Box(
                    x=values[mask == missing], name=name, marker_color=color, legendgroup=name,
                    showlegend=col == 1, boxmean=True, orientation="h", y=[short] * int((mask == missing).sum()),
                    hovertemplate="%{x}<extra></extra>",
                ),
                row=1, col=col,
            )
    fig.update_xaxes(title_text=driver_meta["unit"], row=1, col=1)
    fig.update_xaxes(title_text=target_meta["unit"], row=1, col=2)
    fig.update_yaxes(title_text="Linhas", row=1, col=1)
    _layout(
        fig, f"Simulação {mechanism}: quem ficou ausente?", height=440,
        subtitle="Boxplots (linha tracejada = média) comparando as linhas ausentes e as observadas",
    )
    return fig
