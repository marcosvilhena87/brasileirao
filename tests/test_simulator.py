import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import numpy as np
from brasileirao import simulator


def test_parse_matches():
    df = simulator.parse_matches('data/Brasileirao2025A.txt')
    assert len(df) == 380
    assert {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(df.columns)


def test_league_table():
    df = simulator.parse_matches('data/Brasileirao2025A.txt')
    table = simulator.league_table(df)
    # after first rounds some teams have points
    assert 'points' in table.columns
    assert table['played'].max() > 0


def test_simulate_chances():
    df = simulator.parse_matches('data/Brasileirao2025A.txt')
    chances = simulator.simulate_chances(df, iterations=10)
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_poisson():
    df = simulator.parse_matches('data/Brasileirao2025A.txt')
    chances = simulator.simulate_chances(df, iterations=10, rating_method="poisson")
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_seed_repeatability():
    df = simulator.parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(1234)
    chances1 = simulator.simulate_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(1234)
    chances2 = simulator.simulate_chances(df, iterations=5, rng=rng)
    assert chances1 == chances2


def test_estimate_strengths():
    df = simulator.parse_matches('data/Brasileirao2025A.txt')
    strengths, _, _ = simulator._estimate_strengths(df)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())

    # every team from the matches should appear in the strengths dict
    assert set(teams) == set(strengths.keys())

    # all estimated attack and defense values must be positive
    assert all(v["attack"] > 0 and v["defense"] > 0 for v in strengths.values())
