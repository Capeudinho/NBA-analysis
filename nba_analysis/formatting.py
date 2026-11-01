"""Number formatting in the Brazilian convention (comma as decimal separator)."""

import math

import numpy as np
import pandas as pd

from . import config

MONTHS_PT = {
    1: "jan",
    2: "fev",
    3: "mar",
    4: "abr",
    5: "mai",
    6: "jun",
    7: "jul",
    8: "ago",
    9: "set",
    10: "out",
    11: "nov",
    12: "dez",
}
WEEKDAYS_PT = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def number(value, decimals=config.DECIMALS):
    """1234.567 -> '1.234,57'."""
    if _is_missing(value):
        return "—"
    if isinstance(value, (float, np.floating)) and math.isinf(value):
        return "∞" if value > 0 else "−∞"
    text = f"{float(value):,.{decimals}f}"
    text = text.replace(",", "X").replace(".", ",").replace("X", ".")
    if text.startswith("-"):
        text = "−" + text[1:]
    if text in ("−0", "−0," + "0" * decimals):
        text = text[1:]
    return text


def integer(value):
    if _is_missing(value):
        return "—"
    return number(int(round(float(value))), 0)


def percent(value, decimals=config.DECIMALS):
    """Value already expressed in percent: 12.345 -> '12,35%'."""
    if _is_missing(value):
        return "—"
    return f"{number(value, decimals)}%"


def p_value(value):
    if _is_missing(value):
        return "—"
    if value < 10 ** (-config.P_VALUE_DECIMALS):
        return "< " + number(10 ** (-config.P_VALUE_DECIMALS), config.P_VALUE_DECIMALS)
    return number(value, config.P_VALUE_DECIMALS)


def date(value):
    if _is_missing(value):
        return "—"
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def signed(value, decimals=config.DECIMALS):
    if _is_missing(value):
        return "—"
    text = number(value, decimals)
    return text if text.startswith("−") or float(value) == 0 else "+" + text


def bytes_size(value):
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{number(value / 1024)} KB"
    return f"{number(value / 1024**2)} MB"


def register_filters(app):
    app.jinja_env.filters["num"] = number
    app.jinja_env.filters["int"] = integer
    app.jinja_env.filters["pct"] = percent
    app.jinja_env.filters["pval"] = p_value
    app.jinja_env.filters["date"] = date
    app.jinja_env.filters["signed"] = signed
    app.jinja_env.filters["bytes"] = bytes_size
