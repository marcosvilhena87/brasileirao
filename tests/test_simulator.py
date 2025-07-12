import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
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
