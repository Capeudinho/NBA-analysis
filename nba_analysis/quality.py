"""Special values (zeros, negatives, duplicates, out-of-domain) and consistency rules."""

from functools import lru_cache

import numpy as np
import pandas as pd

from .data import get_dataset, load_raw
from .dictionary import BY_NEW_NAME

PCT_TOLERANCE = 0.0005 + 1e-9  # percentages are rounded to 3 decimals in the source
SEASON_START = pd.Timestamp("2025-10-01")
SEASON_END = pd.Timestamp("2026-04-30")

# Plausible domain of each numeric column (inclusive bounds).
DOMAINS = {
    "season_wins": (0, 82),
    "season_losses": (0, 82),
    "season_win_pct": (0, 1),
    "minutes": (240, 390),
    "field_goals_made": (0, 100),
    "field_goals_attempted": (0, 150),
    "field_goal_pct": (0, 1),
    "three_pointers_made": (0, 50),
    "three_pointers_attempted": (0, 100),
    "three_point_pct": (0, 1),
    "free_throws_made": (0, 80),
    "free_throws_attempted": (0, 100),
    "free_throw_pct": (0, 1),
    "offensive_rebounds": (0, 50),
    "defensive_rebounds": (0, 80),
    "rebounds": (0, 100),
    "assists": (0, 70),
    "steals": (0, 40),
    "blocks": (0, 40),
    "turnovers": (0, 50),
    "personal_fouls": (0, 60),
    "points": (0, 200),
}

ZERO_INTERPRETATION = {
    "win": "Legítimo: 0 codifica derrota (variável binária).",
    "season_wins": "Legítimo: times que ainda não venceram no início da temporada.",
    "season_losses": "Legítimo: times que ainda não perderam no início da temporada.",
    "season_win_pct": "Legítimo: aproveitamento 0 enquanto o time não tem vitórias.",
    "steals": "Raro, mas possível: jogo sem nenhum roubo de bola.",
    "blocks": "Plausível: jogos sem tocos acontecem em times com poucos pivôs protetores de aro.",
    "free_throws_attempted": "Possível em tese (jogo sem lances livres), mas muito raro.",
}


@lru_cache(maxsize=1)
def special_values():
    raw = load_raw()
    df = get_dataset()
    rows = []
    for column in ["win"] + list(DOMAINS):
        series = df[column]
        low, high = DOMAINS.get(column, (0, 1))
        zeros = int((series == 0).sum())
        rows.append(
            {
                "column": column,
                "label": BY_NEW_NAME[column]["label"],
                "zeros": zeros,
                "zeros_pct": zeros / len(series) * 100,
                "negatives": int((series < 0).sum()),
                "domain": (low, high),
                "out_of_domain": int(((series < low) | (series > high)).sum()),
                "interpretation": ZERO_INTERPRETATION.get(column, "Sem zeros.") if zeros else "Sem zeros.",
            }
        )
    duplicates = {
        "full_rows": int(raw.duplicated().sum()),
        "team_game_key": int(raw.duplicated(["Team_ID", "Game_ID"]).sum()),
        "stats_without_ids": int(
            raw.drop(columns=["Team_ID", "Game_ID", "TEAM_NAME"]).duplicated().sum()
        ),
    }
    return rows, duplicates


def _rule(name, description, violations, total):
    return {
        "name": name,
        "description": description,
        "violations": int(violations),
        "total": int(total),
        "ok": int(violations) == 0,
    }


@lru_cache(maxsize=1)
def consistency_rules():
    df = get_dataset()
    n = len(df)
    games = df.groupby("game_id")
    n_games = games.ngroups
    rules = []

    rules.append(_rule(
        "Dois registros por partida",
        "Cada game_id deve aparecer exatamente 2 vezes (um registro por time).",
        (games.size() != 2).sum(), n_games,
    ))
    rules.append(_rule(
        "Um vencedor por partida",
        "Em cada partida, a soma de win deve ser 1 (um time vence e o outro perde).",
        (games["win"].sum() != 1).sum(), n_games,
    ))
    rules.append(_rule(
        "Mesma duração para os dois times",
        "Os dois registros da partida devem ter os mesmos minutos (mesmas prorrogações).",
        (games["minutes"].nunique() != 1).sum(), n_games,
    ))
    rules.append(_rule(
        "Mesma data para os dois times",
        "Os dois registros da partida devem ter a mesma data.",
        (games["game_date"].nunique() != 1).sum(), n_games,
    ))
    pairs = df[["game_id", "team", "opponent", "is_home"]]
    merged = pairs.merge(pairs, on="game_id", suffixes=("", "_other"))
    merged = merged[merged["team"] != merged["team_other"]]
    mirror_bad = (merged["opponent"] != merged["team_other"]) | (merged["is_home"] + merged["is_home_other"] != 1)
    rules.append(_rule(
        "Confronto espelhado",
        "Se A recebe B (\"A vs. B\"), o outro registro deve ser \"B @ A\".",
        mirror_bad.sum(), len(merged),
    ))
    rules.append(_rule(
        "Um jogo por dia por time",
        "Um time não pode ter dois jogos na mesma data.",
        df.duplicated(["team", "game_date"]).sum(), n,
    ))
    rules.append(_rule(
        "Sigla × nome × ID coerentes",
        "Cada team_id deve corresponder a uma única sigla e a um único nome.",
        (df.groupby("team_id")[["team", "team_name"]].nunique() > 1).any(axis=1).sum(),
        df["team_id"].nunique(),
    ))
    for made, attempted, label in (
        ("field_goals_made", "field_goals_attempted", "FGM ≤ FGA"),
        ("three_pointers_made", "three_pointers_attempted", "FG3M ≤ FG3A"),
        ("free_throws_made", "free_throws_attempted", "FTM ≤ FTA"),
        ("three_pointers_made", "field_goals_made", "FG3M ≤ FGM"),
        ("three_pointers_attempted", "field_goals_attempted", "FG3A ≤ FGA"),
    ):
        rules.append(_rule(label, "Convertidos não podem exceder tentados (e cestas de 3 são parte das de quadra).",
                           (df[made] > df[attempted]).sum(), n))
    for pct, made, attempted, label in (
        ("field_goal_pct", "field_goals_made", "field_goals_attempted", "FG_PCT = FGM / FGA"),
        ("three_point_pct", "three_pointers_made", "three_pointers_attempted", "FG3_PCT = FG3M / FG3A"),
        ("free_throw_pct", "free_throws_made", "free_throws_attempted", "FT_PCT = FTM / FTA"),
    ):
        ratio = (df[made] / df[attempted].replace(0, np.nan)).fillna(0)
        rules.append(_rule(label, "Percentual deve coincidir com a razão (tolerância de arredondamento 0,0005).",
                           ((df[pct] - ratio).abs() > PCT_TOLERANCE).sum(), n))
    rules.append(_rule(
        "REB = OREB + DREB", "Rebotes totais devem ser a soma de ofensivos e defensivos.",
        (df["rebounds"] != df["offensive_rebounds"] + df["defensive_rebounds"]).sum(), n,
    ))
    rules.append(_rule(
        "PTS = 2·FGM + FG3M + FTM", "Pontos devem fechar com cestas de quadra, de 3 e lances livres.",
        (df["points"] != 2 * df["field_goals_made"] + df["three_pointers_made"] + df["free_throws_made"]).sum(), n,
    ))
    rules.append(_rule(
        "MIN = 240 + 25·k", "Minutos do time: 240 no tempo normal e +25 por prorrogação.",
        (((df["minutes"] - 240) % 25 != 0) | (df["minutes"] < 240)).sum(), n,
    ))
    rules.append(_rule(
        "W + L = nº do jogo", "Vitórias + derrotas acumuladas devem igualar a ordem do jogo na temporada.",
        (df["season_wins"] + df["season_losses"] != df["game_number"]).sum(), n,
    ))
    previous_wins = df.groupby("team_id")["season_wins"].shift(fill_value=0)
    rules.append(_rule(
        "W acumulado coerente com WL", "W aumenta em 1 exatamente quando o jogo é uma vitória.",
        ((df["season_wins"] - previous_wins) != df["win"]).sum(), n,
    ))
    expected_pct = df["season_wins"] / (df["season_wins"] + df["season_losses"])
    rules.append(_rule(
        "W_PCT = W / (W + L)", "Aproveitamento acumulado coerente com W e L.",
        ((df["season_win_pct"] - expected_pct).abs() > PCT_TOLERANCE).sum(), n,
    ))
    rules.append(_rule(
        "Datas válidas e na temporada regular",
        "Datas convertíveis e entre 01/10/2025 e 30/04/2026.",
        (df["game_date"].isna() | (df["game_date"] < SEASON_START) | (df["game_date"] > SEASON_END)).sum(), n,
    ))
    return rules


def games_per_team():
    df = get_dataset()
    counts = df.groupby("team").size()
    return {"min": int(counts.min()), "max": int(counts.max()), "teams": int(counts.size)}
