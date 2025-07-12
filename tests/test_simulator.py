import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import numpy as np
from brasileirao import parse_matches, league_table, simulate_chances
from brasileirao import simulator


def test_parse_matches():
    df = parse_matches('data/Brasileirao2025A.txt')
    assert len(df) == 380
    assert {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(df.columns)


def test_league_table():
    df = parse_matches('data/Brasileirao2025A.txt')
    table = league_table(df)
    # after first rounds some teams have points
    assert 'points' in table.columns
    assert table['played'].max() > 0


def test_league_table_deterministic_sorting():
    data = [
        {
            'date': '2025-01-01',
            'home_team': 'Alpha',
            'away_team': 'Beta',
            'home_score': 1,
            'away_score': 0,
        },
        {
            'date': '2025-01-02',
            'home_team': 'Beta',
            'away_team': 'Gamma',
            'home_score': 1,
            'away_score': 0,
        },
        {
            'date': '2025-01-03',
            'home_team': 'Gamma',
            'away_team': 'Alpha',
            'home_score': 1,
            'away_score': 0,
        },
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team) == sorted(table.team)


def test_simulate_chances():
    df = parse_matches('data/Brasileirao2025A.txt')
    chances = simulate_chances(df, iterations=10)
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_poisson():
    df = parse_matches('data/Brasileirao2025A.txt')
    chances = simulate_chances(df, iterations=10, rating_method="poisson")
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(1234)
    chances1 = simulate_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(1234)
    chances2 = simulate_chances(df, iterations=5, rng=rng)
    assert chances1 == chances2


def test_estimate_strengths():
    df = parse_matches('data/Brasileirao2025A.txt')
    strengths, _, _ = simulator._estimate_strengths(df)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())

    # every team from the matches should appear in the strengths dict
    assert set(teams) == set(strengths.keys())

    # all estimated attack and defense values must be positive
    assert all(v["attack"] > 0 and v["defense"] > 0 for v in strengths.values())


def test_estimate_strengths_with_history():
    df = parse_matches('data/Brasileirao2025A.txt')
    strengths, _, _ = simulator.estimate_strengths_with_history(df)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    assert set(teams).issubset(set(strengths.keys()))


def test_simulate_chances_historic_ratio():
    df = parse_matches('data/Brasileirao2025A.txt')
    chances = simulate_chances(df, iterations=10, rating_method="historic_ratio")
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_elo_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(42)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=15.0,
    )
    rng = np.random.default_rng(42)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=15.0,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_elo_k_value_changes_results():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(99)
    chances_low = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=5.0,
    )
    rng = np.random.default_rng(99)
    chances_high = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=40.0,
    )
    assert chances_low != chances_high


def test_simulate_chances_neg_binom_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(7)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="neg_binom",
        rng=rng,
    )
    rng = np.random.default_rng(7)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="neg_binom",
        rng=rng,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_team_home_advantage_changes_results():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(11)
    base = simulate_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(11)
    custom = simulate_chances(
        df,
        iterations=5,
        rng=rng,
        team_home_advantages={"Bahia": 2.0},
    )
    assert base != custom


def test_simulate_chances_dixon_coles_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(123)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="dixon_coles",
        rng=rng,
    )
    rng = np.random.default_rng(123)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="dixon_coles",
        rng=rng,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_compute_leader_stats():
    data = [
        {
            "date": "2025-01-01",
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 1,
            "away_score": 0,
        },
        {
            "date": "2025-01-02",
            "home_team": "Alpha",
            "away_team": "Gamma",
            "home_score": 0,
            "away_score": 1,
        },
        {
            "date": "2025-01-03",
            "home_team": "Beta",
            "away_team": "Gamma",
            "home_score": 2,
            "away_score": 0,
        },
    ]
    df = pd.DataFrame(data)
    counts = simulator.compute_leader_stats(df)
    assert counts["Alpha"] == 1
    assert counts["Beta"] == 1
    assert counts["Gamma"] == 1


def _slow_leader_stats(df: pd.DataFrame) -> dict:
    """Naive reference implementation using league_table."""
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    counts = {t: 0 for t in teams}
    played: list[dict] = []
    for _, row in df.sort_values("date").iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        played.append(row.to_dict())
        table = simulator.league_table(pd.DataFrame(played))
        counts[table.iloc[0]["team"]] += 1
    return counts


def test_compute_leader_stats_equivalence():
    df = parse_matches("data/Brasileirao2025A.txt")
    assert simulator.compute_leader_stats(df) == _slow_leader_stats(df)


def test_simulate_chances_leader_history_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(55)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="leader_history",
        rng=rng,
        leader_history_paths=["data/Brasileirao2024A.txt"],
        leader_history_weight=0.5,
    )
    rng = np.random.default_rng(55)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="leader_history",
        rng=rng,
        leader_history_paths=["data/Brasileirao2024A.txt"],
        leader_history_weight=0.5,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6
