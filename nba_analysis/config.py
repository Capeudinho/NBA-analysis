"""Project-wide constants: file paths, teams, variables, labels and thresholds."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "stats_times_regular_final.csv"

# Number formatting convention used by every table in the interface.
DECIMALS = 2
P_VALUE_DECIMALS = 4

# Teams assigned to the group (abbreviation -> metadata).
TEAMS = {
    "NYK": {"name": "New York Knicks", "short": "Knicks", "color": "#2a78d6"},
    "PHX": {"name": "Phoenix Suns", "short": "Suns", "color": "#eb6834"},
}
TEAM_CODES = list(TEAMS)
LEAGUE_CODE = "NBA"
LEAGUE_LABEL = "Liga (todos os 30 times)"
LEAGUE_COLOR = "#8a8984"

# Variables required by the descriptive-statistics section of the assignment.
# Keys are the renamed (intuitive) column names used throughout the code.
ANALYSIS_VARIABLES = {
    "points": {
        "label": "Pontos",
        "abbr": "PTS",
        "unit": "pontos por jogo",
        "unit_short": "pts",
    },
    "rebounds": {
        "label": "Rebotes",
        "abbr": "REB",
        "unit": "rebotes por jogo",
        "unit_short": "reb",
    },
    "assists": {
        "label": "Assistências",
        "abbr": "AST",
        "unit": "assistências por jogo",
        "unit_short": "ast",
    },
    "steals": {
        "label": "Roubos de bola",
        "abbr": "STL",
        "unit": "roubos de bola por jogo",
        "unit_short": "stl",
    },
    "blocks": {
        "label": "Tocos",
        "abbr": "BLK",
        "unit": "tocos por jogo",
        "unit_short": "blk",
    },
    "personal_fouls": {
        "label": "Faltas pessoais",
        "abbr": "PF",
        "unit": "faltas pessoais por jogo",
        "unit_short": "pf",
    },
    "three_pointers_made": {
        "label": "Cestas de 3 convertidas",
        "abbr": "FG3M",
        "unit": "cestas de 3 pontos por jogo",
        "unit_short": "3pm",
    },
}
VARIABLE_KEYS = list(ANALYSIS_VARIABLES)

# Outlier / robustness parameters (documented in the interface).
IQR_K = 1.5
IQR_K_EXTREME = 3.0
Z_THRESHOLD = 3.0
GRUBBS_ALPHA = 0.05
KNN_K = 10
LOF_NEIGHBORS = 20
SIGNIFICANCE = 0.05
RANDOM_SEED = 42

# Skewness intensity thresholds (Bulmer, 1979).
SKEW_MODERATE = 0.5
SKEW_STRONG = 1.0

# Means below this value are flagged as "small" when judging CV validity.
CV_SMALL_MEAN = 10.0
