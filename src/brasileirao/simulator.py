from __future__ import annotations

import numpy as np
import pandas as pd
import re
from pathlib import Path


def _parse_date(date_str: str) -> pd.Timestamp:
    """Parse dates from multiple formats."""
    parts = date_str.split("/")
    year = parts[-1]
    if len(year) == 4:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    return pd.to_datetime(date_str, format="%m/%d/%y")

SCORE_PATTERN = re.compile(r"(\d+/\d+/\d+)\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s*(?:\(ID:.*)?$")
NOSCORE_PATTERN = re.compile(r"(\d+/\d+/\d+)\s+(.+?)\s{2,}(.+?)\s*(?:\(ID:.*)?$")


def parse_matches(path: str | Path) -> pd.DataFrame:
    """Parse the fixture text file into a DataFrame."""
    rows: list[dict] = []
    in_games = False
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == 'GamesBegin':
                in_games = True
                continue
            if line.strip() == 'GamesEnd':
                break
            if not in_games:
                continue
            line = line.rstrip('\n')
            m = SCORE_PATTERN.match(line)
            if m:
                date_str, home, hs, as_, away = m.groups()
                rows.append({
                    'date': _parse_date(date_str),
                    'home_team': home.strip(),
                    'away_team': away.strip(),
                    'home_score': int(hs),
                    'away_score': int(as_),
                })
                continue
            m = NOSCORE_PATTERN.match(line)
            if m:
                date_str, home, away = m.groups()
                rows.append({
                    'date': _parse_date(date_str),
                    'home_team': home.strip(),
                    'away_team': away.strip(),
                    'home_score': np.nan,
                    'away_score': np.nan,
                })
    return pd.DataFrame(rows)


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute league standings from match results."""
    teams = pd.unique(matches[['home_team', 'away_team']].values.ravel())
    table = {t: {'team': t, 'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0} for t in teams}

    played = matches.dropna(subset=['home_score', 'away_score'])
    for _, row in played.iterrows():
        home = row['home_team']
        away = row['away_team']
        hs = int(row['home_score'])
        as_ = int(row['away_score'])
        table[home]['played'] += 1
        table[away]['played'] += 1
        table[home]['gf'] += hs
        table[home]['ga'] += as_
        table[away]['gf'] += as_
        table[away]['ga'] += hs
        if hs > as_:
            table[home]['wins'] += 1
            table[home]['points'] = table[home].get('points', 0) + 3
            table[away]['losses'] += 1
            table[away].setdefault('points', 0)
        elif hs < as_:
            table[away]['wins'] += 1
            table[away]['points'] = table[away].get('points', 0) + 3
            table[home]['losses'] += 1
            table[home].setdefault('points', 0)
        else:
            table[home]['draws'] += 1
            table[away]['draws'] += 1
            table[home]['points'] = table[home].get('points', 0) + 1
            table[away]['points'] = table[away].get('points', 0) + 1

    for t in table.values():
        t.setdefault('points', 0)
        t['gd'] = t['gf'] - t['ga']

    df = pd.DataFrame(table.values())
    df = df.sort_values(['points', 'gd', 'gf'], ascending=False).reset_index(drop=True)
    return df


def _estimate_strengths(matches: pd.DataFrame):
    played = matches.dropna(subset=['home_score', 'away_score'])
    total_goals = played['home_score'].sum() + played['away_score'].sum()
    total_games = len(played)
    avg_goals = total_goals / total_games if total_games else 2.5
    home_adv = played['home_score'].sum() / played['away_score'].sum() if played['away_score'].sum() else 1.0

    teams = pd.unique(matches[['home_team', 'away_team']].values.ravel())
    strengths = {}
    for team in teams:
        gf = (
            played.loc[played.home_team == team, 'home_score'].sum() +
            played.loc[played.away_team == team, 'away_score'].sum()
        )
        ga = (
            played.loc[played.home_team == team, 'away_score'].sum() +
            played.loc[played.away_team == team, 'home_score'].sum()
        )
        gp = played.loc[(played.home_team == team) | (played.away_team == team)].shape[0]
        if gp == 0:
            attack = defense = 1.0
        else:
            attack = (gf / gp) / avg_goals
            defense = (ga / gp) / avg_goals
        strengths[team] = {'attack': attack, 'defense': defense}
    return strengths, avg_goals, home_adv


def estimate_strengths_with_history(
    current_matches: pd.DataFrame | None = None,
    past_path: str | Path = "data/Brasileirao2024A.txt",
    past_weight: float = 0.5,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Estimate strengths using current season matches and weighted history."""
    if current_matches is None:
        current_matches = parse_matches("data/Brasileirao2025A.txt")
    past_matches = parse_matches(past_path)
    if 0 < past_weight < 1:
        past_matches = past_matches.sample(frac=past_weight, random_state=0).reset_index(drop=True)
    combined = pd.concat([current_matches, past_matches], ignore_index=True)
    return _estimate_strengths(combined)


def estimate_poisson_strengths(matches: pd.DataFrame):
    """Fit a Poisson regression model to estimate team strengths."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    played = matches.dropna(subset=["home_score", "away_score"])

    rows: list[dict] = []
    for _, row in played.iterrows():
        rows.append(
            {
                "team": row["home_team"],
                "opponent": row["away_team"],
                "home": 1,
                "goals": row["home_score"],
            }
        )
        rows.append(
            {
                "team": row["away_team"],
                "opponent": row["home_team"],
                "home": 0,
                "goals": row["away_score"],
            }
        )

    df = pd.DataFrame(rows)

    model = smf.glm(
        "goals ~ home + C(team) + C(opponent)",
        data=df,
        family=sm.families.Poisson(),
    ).fit()

    base_mu = float(np.exp(model.params["Intercept"]))
    home_adv = float(np.exp(model.params.get("home", 0.0)))

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    strengths: dict[str, dict[str, float]] = {}
    for t in teams:
        atk_coef = model.params.get(f"C(team)[T.{t}]", 0.0)
        def_coef = model.params.get(f"C(opponent)[T.{t}]", 0.0)
        strengths[t] = {
            "attack": float(np.exp(atk_coef)),
            "defense": float(np.exp(def_coef)),
        }

    return strengths, base_mu, home_adv


def simulate_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    rating_method: str = "ratio",
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Simulate remaining fixtures and return title probabilities.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame containing all fixtures. Played games must have scores.
    iterations : int, default 1000
        Number of simulation runs.
    rating_method : str, default "ratio"
        Method used to estimate team strengths.
    rng : np.random.Generator | None, optional
        Random number generator to use. A new generator is created when ``None``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rating_method == "poisson":
        strengths, avg_goals, home_adv = estimate_poisson_strengths(matches)
    elif rating_method == "historic_ratio":
        strengths, avg_goals, home_adv = estimate_strengths_with_history(matches)
    else:
        strengths, avg_goals, home_adv = _estimate_strengths(matches)
    teams = pd.unique(matches[['home_team', 'away_team']].values.ravel())
    champs = {t: 0 for t in teams}

    played_df = matches.dropna(subset=['home_score', 'away_score'])
    remaining = matches[matches['home_score'].isna() | matches['away_score'].isna()]

    for _ in range(iterations):
        sims = []
        for _, row in remaining.iterrows():
            ht = row['home_team']
            at = row['away_team']
            mu_home = avg_goals * strengths[ht]['attack'] * strengths[at]['defense'] * home_adv
            mu_away = avg_goals * strengths[at]['attack'] * strengths[ht]['defense']
            hs = rng.poisson(mu_home)
            as_ = rng.poisson(mu_away)
            sims.append({'date': row['date'], 'home_team': ht, 'away_team': at, 'home_score': hs, 'away_score': as_})
        all_matches = pd.concat([played_df, pd.DataFrame(sims)], ignore_index=True)
        table = league_table(all_matches)
        champs[table.iloc[0]['team']] += 1

    for t in champs:
        champs[t] = champs[t] / iterations
    return champs
