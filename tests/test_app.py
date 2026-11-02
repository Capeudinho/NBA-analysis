"""Smoke and correctness tests. Run with: python -m pytest"""

import numpy as np
import pandas as pd
import pytest

from nba_analysis import config, create_app, missing, outliers, quality, stats
from nba_analysis.data import get_dataset, load_raw
from nba_analysis.formatting import number, p_value, percent

PAGES = [
    "/",
    "/base",
    "/dicionario",
    "/qualidade/especiais",
    "/qualidade/ausentes",
    "/qualidade/ausentes?mechanism=MNAR&target=blocks&rate=0.05&seed=3",
    "/qualidade/outliers",
    "/qualidade/outliers?scope=PHX&column=minutes&k=3",
    "/descritiva",
    "/descritiva?variable=steals&team=PHX&scale=z",
    "/assimetria",
    "/cv",
    "/iqr",
    "/sintese",
    "/descritiva?variable=invalid&team=invalid",
]


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.mark.parametrize("url", PAGES)
def test_pages_render(client, url):
    response = client.get(url)
    assert response.status_code == 200
    assert "<html lang=\"pt-BR\">" in response.get_data(as_text=True)


def test_treatment():
    df = get_dataset()
    assert len(df) == len(load_raw())
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"])
    assert set(df["win"].unique()) <= {0, 1}
    assert set(df["is_home"].unique()) == {0, 1}
    assert df["team"].isna().sum() == 0
    assert {"points", "rebounds", "three_pointers_made"} <= set(df.columns)


def test_win_loss_text_is_converted():
    from nba_analysis.data import _encode_win_loss

    encoded, how = _encode_win_loss(pd.Series(["W", "L", " w "]))
    assert how == "mapped"
    assert encoded.tolist() == [1, 0, 1]


def test_describe_matches_pandas():
    values = get_dataset().query("team == 'NYK'")["points"]
    d = stats.describe(values)
    assert d["mean"] == pytest.approx(values.mean())
    assert d["median"] == pytest.approx(values.median())
    assert d["std"] == pytest.approx(values.std())
    assert d["var"] == pytest.approx(values.var())
    assert d["q1"] == pytest.approx(values.quantile(0.25))
    assert d["q3"] == pytest.approx(values.quantile(0.75))


def test_skewness_matches_pandas():
    values = get_dataset().query("team == 'PHX'")["rebounds"]
    assert stats.skewness(values)["G1"] == pytest.approx(values.skew())


def test_iqr_outliers_match_boxplot():
    summary = stats.team_variable_summary()
    for variable in config.VARIABLE_KEYS:
        for team in config.TEAM_CODES:
            assert summary[variable][team]["iqr"]["matches_boxplot"]


def test_consistency_rules_pass():
    assert all(rule["ok"] for rule in quality.consistency_rules())


def test_grubbs_finds_planted_outlier():
    rng = np.random.default_rng(0)
    x = np.append(rng.normal(100, 5, 80), 160)
    mask, _ = outliers.grubbs_outliers(x)
    assert mask[-1] and mask.sum() == 1


def test_missing_tests():
    assert not missing.real_data_tests()["little"]["applicable"]
    mar = missing.run_simulation("MAR", "points", "assists", 0.2, 42)
    assert mar["little"]["p_value"] < 0.05
    assert mar["fairclough"]["models"][0]["reject"]


def test_health_route(client):
    response = client.get("/saude")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["linhas"] == len(get_dataset())
    assert body["times_do_grupo"] == config.TEAM_CODES


def test_number_formatting():
    assert number(1234.567) == "1.234,57"
    assert number(-0.5) == "−0,50"
    assert percent(12.346) == "12,35%"
    assert p_value(0.00001) == "< 0,0001"
