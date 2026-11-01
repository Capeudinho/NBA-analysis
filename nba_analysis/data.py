"""Loading and treatment of the dataset (renaming, type conversion, date handling)."""

from functools import lru_cache

import pandas as pd

from . import config
from .dictionary import COLUMNS, RENAME_MAP

WIN_LOSS_MAP = {"W": 1, "L": 0}
REST_DAYS_CAP = 5


@lru_cache(maxsize=1)
def load_raw():
    """Read the CSV exactly as delivered (no treatment)."""
    return pd.read_csv(config.DATA_PATH)


def _encode_win_loss(series):
    """Convert the WL column to 0/1, accepting either 'W'/'L' strings or numbers."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64"), "numeric"
    mapped = series.astype(str).str.strip().str.upper().map(WIN_LOSS_MAP)
    return mapped.astype("int64"), "mapped"


def _parse_matchup(matchup):
    """Split 'NYK vs. CLE' / 'NYK @ BOS' into team, opponent and home flag."""
    parts = matchup.str.extract(r"^(?P<team>[A-Z]{3})\s+(?P<sep>vs\.|@)\s+(?P<opponent>[A-Z]{3})$")
    is_home = (parts["sep"] == "vs.").astype("int64")
    return parts["team"], parts["opponent"], is_home


def prepare(raw):
    """Apply every treatment step and return (treated DataFrame, treatment log)."""
    log = {}
    df = raw.rename(columns=RENAME_MAP).copy()

    # Date: string -> datetime64. Invalid dates would become NaT and be counted.
    log["date_dtype_before"] = str(raw["GAME_DATE"].dtype)
    df["game_date"] = pd.to_datetime(df["game_date"], format="%Y-%m-%d", errors="coerce")
    log["invalid_dates"] = int(df["game_date"].isna().sum())
    log["date_dtype_after"] = str(df["game_date"].dtype)
    log["date_min"] = df["game_date"].min()
    log["date_max"] = df["game_date"].max()

    # Win/loss: categorical -> numeric binary.
    log["wl_values_before"] = sorted(map(str, raw["WL"].dropna().unique()))
    df["win"], log["wl_conversion"] = _encode_win_loss(df["win"])

    # Cumulative W and L: make sure they are integers.
    for column in ("season_wins", "season_losses"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("int64")

    # Matchup text -> team abbreviation, opponent and home flag.
    df["team"], df["opponent"], df["is_home"] = _parse_matchup(df["matchup"])
    log["unparsed_matchups"] = int(df["team"].isna().sum())

    # Date-derived features.
    df = df.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    df["game_month"] = df["game_date"].dt.month.astype("int64")
    df["game_weekday"] = df["game_date"].dt.day_of_week.astype("int64")
    df["game_number"] = df.groupby("team_id").cumcount().add(1).astype("int64")
    gap = df.groupby("team_id")["game_date"].diff().dt.days
    df["days_since_previous_game"] = (
        gap.fillna(REST_DAYS_CAP).clip(upper=REST_DAYS_CAP).astype("int64")
    )
    df["is_back_to_back"] = (gap == 1).astype("int64")

    # Overtime periods derived from total minutes.
    df["overtime_periods"] = ((df["minutes"] - 240) // 25).astype("int64")

    log["rows"], log["columns_original"] = raw.shape
    log["columns_treated"] = df.shape[1]
    return df, log


@lru_cache(maxsize=1)
def get_dataset():
    """Treated dataset (cached)."""
    df, _ = prepare(load_raw())
    return df


@lru_cache(maxsize=1)
def get_treatment_log():
    _, log = prepare(load_raw())
    return log


def ml_ready(df):
    """Dataset restricted to the columns kept for machine-learning models."""
    from .dictionary import DERIVED_COLUMNS, DROP

    keep = [c["new"] for c in COLUMNS + DERIVED_COLUMNS if c["ml"] != DROP]
    return df[keep]


def team_data(team_code):
    """Rows of a single team (by abbreviation) or the whole league."""
    df = get_dataset()
    if team_code in (None, "", config.LEAGUE_CODE):
        return df
    return df[df["team"] == team_code]


def numeric_quality_columns(df):
    """Numeric columns relevant for quality checks (identifiers and binaries excluded)."""
    excluded = {"team_id", "game_id", "win", "is_home", "is_back_to_back"}
    original_new_names = [c["new"] for c in COLUMNS]
    return [
        c
        for c in original_new_names
        if c in df.columns and c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def file_size_bytes():
    return config.DATA_PATH.stat().st_size


def memory_usage_bytes(df):
    return int(df.memory_usage(deep=True).sum())


def semantic_type_counts():
    counts = {}
    for col in COLUMNS:
        counts[col["semantic_type"]] = counts.get(col["semantic_type"], 0) + 1
    return counts
