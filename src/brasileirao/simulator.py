"""Utilities for parsing match fixtures and running Monte Carlo simulations.

This module reads fixtures exported from SportsClubStats and provides
functions to compute league tables and run a SportsClubStats-style model to
project results.  It powers the public functions exposed in
``brasileirao.__init__``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
SCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s*(?:\(ID:.*)?$"
)
NOSCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s{2,}(.+?)\s*(?:\(ID:.*)?$"
)


def _parse_date(date_str: str) -> pd.Timestamp:
    parts = date_str.split("/")
    year = parts[-1]
    if len(year) == 4:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    return pd.to_datetime(date_str, format="%m/%d/%y")


def parse_matches(path: str | Path) -> pd.DataFrame:
    """Return a DataFrame of fixtures and results."""
    rows: list[dict] = []
    in_games = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "GamesBegin":
                in_games = True
                continue
            if line.strip() == "GamesEnd":
                break
            if not in_games:
                continue
            line = line.rstrip("\n")
            m = SCORE_PATTERN.match(line)
            if m:
                date_str, home, hs, as_, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": int(hs),
                        "away_score": int(as_),
                    }
                )
                continue
            m = NOSCORE_PATTERN.match(line)
            if m:
                date_str, home, away = m.groups()
                rows.append(
                    {
                        "date": _parse_date(date_str),
                        "home_team": home.strip(),
                        "away_team": away.strip(),
                        "home_score": np.nan,
                        "away_score": np.nan,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table computation
# ---------------------------------------------------------------------------

def _head_to_head_points(matches: pd.DataFrame, teams: list[str]) -> Dict[str, int]:
    points = {t: 0 for t in teams}
    df = matches.dropna(subset=["home_score", "away_score"])
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)]
    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        if hs > as_:
            points[ht] += 3
        elif hs < as_:
            points[at] += 3
        else:
            points[ht] += 1
            points[at] += 1
    return points


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute league standings from played matches."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    table: Dict[str, Dict[str, float]] = {
        t: {"team": t, "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
        for t in teams
    }

    played = matches.dropna(subset=["home_score", "away_score"])
    for _, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["gf"] += hs
        table[home]["ga"] += as_
        table[away]["gf"] += as_
        table[away]["ga"] += hs
        if hs > as_:
            table[home]["wins"] += 1
            table[home]["points"] = table[home].get("points", 0) + 3
            table[away]["losses"] += 1
            table[away].setdefault("points", 0)
        elif hs < as_:
            table[away]["wins"] += 1
            table[away]["points"] = table[away].get("points", 0) + 3
            table[home]["losses"] += 1
            table[home].setdefault("points", 0)
        else:
            table[home]["draws"] += 1
            table[away]["draws"] += 1
            table[home]["points"] = table[home].get("points", 0) + 1
            table[away]["points"] = table[away].get("points", 0) + 1

    for t in table.values():
        t.setdefault("points", 0)
        t["gd"] = t["gf"] - t["ga"]

    df = pd.DataFrame(table.values())
    df["head_to_head"] = 0
    for _, group in df.groupby(["points", "wins", "gd", "gf"]):
        if len(group) <= 1:
            continue
        teams_tied = group["team"].tolist()
        h2h = _head_to_head_points(played, teams_tied)
        for t, val in h2h.items():
            df.loc[df["team"] == t, "head_to_head"] = val

    df = df.sort_values(
        ["points", "wins", "gd", "gf", "head_to_head", "team"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Rating model (SportsClubStats-style)
# ---------------------------------------------------------------------------

def _estimate_strengths(
    matches: pd.DataFrame,
    smooth: float = 1.0,
    *,
    avg_goals_baseline: float = 2.5,
    home_adv_baseline: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    played = matches.dropna(subset=["home_score", "away_score"])
    total_goals = played["home_score"].sum() + played["away_score"].sum()
    total_games = len(played)
    avg_goals = total_goals / total_games if total_games else avg_goals_baseline
    away_total = played["away_score"].sum()
    if away_total:
        home_adv = played["home_score"].sum() / away_total
    else:
        home_adv = home_adv_baseline

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    strengths: Dict[str, Dict[str, float]] = {}
    for team in teams:
        gf = (
            played.loc[played.home_team == team, "home_score"].sum()
            + played.loc[played.away_team == team, "away_score"].sum()
        )
        ga = (
            played.loc[played.home_team == team, "away_score"].sum()
            + played.loc[played.away_team == team, "home_score"].sum()
        )
        gp = played.loc[(played.home_team == team) | (played.away_team == team)].shape[0]
        if gp == 0:
            attack = defense = 1.0
        else:
            attack = ((gf + smooth) / (gp + smooth)) / avg_goals
            defense = ((ga + smooth) / (gp + smooth)) / avg_goals
        strengths[team] = {"attack": attack, "defense": defense}
    return strengths, avg_goals, home_adv


def _estimate_team_home_advantages(
    matches: pd.DataFrame,
    smooth: float = 0.0,
    *,
    baseline: float | None = None,
) -> Dict[str, float]:
    played = matches.dropna(subset=["home_score", "away_score"])
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())

    if baseline is None:
        total_home = played["home_score"].sum()
        total_away = played["away_score"].sum()
        baseline = total_home / total_away if total_away else 1.0

    factors: Dict[str, float] = {}
    for t in teams:
        home_games = played[played.home_team == t]
        away_games = played[played.away_team == t]
        if not len(home_games) or not len(away_games):
            factors[t] = 1.0
            continue
        home_gpg = (home_games["home_score"].sum() + smooth) / (len(home_games) + smooth)
        away_gpg = (away_games["away_score"].sum() + smooth) / (len(away_games) + smooth)
        if away_gpg == 0 or np.isnan(home_gpg) or np.isnan(away_gpg):
            factors[t] = 1.0
        else:
            factors[t] = float((home_gpg / away_gpg) / baseline)
    return factors


def _prepare_team_home_advantages(
    matches: pd.DataFrame,
    custom: Dict[str, float] | None,
    *,
    smooth: float = 0.0,
    baseline: float | None = None,
) -> Dict[str, float]:
    base = _estimate_team_home_advantages(matches, smooth=smooth, baseline=baseline)
    if custom:
        base.update(custom)
    return base


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_table(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    strengths: Dict[str, Dict[str, float]],
    avg_goals: float,
    home_adv: float,
    team_home_advantages: Dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    sims: list[dict] = []
    for _, row in remaining.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        factor = team_home_advantages.get(ht, 1.0)
        mu_home = (
            avg_goals
            * strengths[ht]["attack"]
            * strengths[at]["defense"]
            * home_adv
            * factor
        )
        mu_away = avg_goals * strengths[at]["attack"] * strengths[ht]["defense"]
        hs = rng.poisson(mu_home)
        as_ = rng.poisson(mu_away)
        sims.append(
            {
                "date": row["date"],
                "home_team": ht,
                "away_team": at,
                "home_score": hs,
                "away_score": as_,
            }
        )
    all_matches = pd.concat([played_df, pd.DataFrame(sims)], ignore_index=True)
    return league_table(all_matches)


# ---------------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------------

def simulate_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    team_home_advantages: Dict[str, float] | None = None,
    smooth: float = 1.0,
    avg_goals_baseline: float = 2.5,
    home_adv_baseline: float = 1.0,
    home_smooth: float = 0.0,
    home_baseline: float | None = None,
) -> Dict[str, float]:
    """Return title probabilities using a SportsClubStats-style model."""
    if rng is None:
        rng = np.random.default_rng()

    team_home_advantages = _prepare_team_home_advantages(
        matches,
        team_home_advantages,
        smooth=home_smooth,
        baseline=home_baseline,
    )
    strengths, avg_goals, home_adv = _estimate_strengths(
        matches,
        smooth=smooth,
        avg_goals_baseline=avg_goals_baseline,
        home_adv_baseline=home_adv_baseline,
    )

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    champs = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            strengths,
            avg_goals,
            home_adv,
            team_home_advantages,
            rng,
        )
        champs[table.iloc[0]["team"]] += 1

    for t in champs:
        champs[t] /= iterations
    return champs


def simulate_relegation_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    team_home_advantages: Dict[str, float] | None = None,
    smooth: float = 1.0,
    avg_goals_baseline: float = 2.5,
    home_adv_baseline: float = 1.0,
    home_smooth: float = 0.0,
    home_baseline: float | None = None,
) -> Dict[str, float]:
    """Return probabilities of finishing in the bottom four."""
    if rng is None:
        rng = np.random.default_rng()

    team_home_advantages = _prepare_team_home_advantages(
        matches,
        team_home_advantages,
        smooth=home_smooth,
        baseline=home_baseline,
    )
    strengths, avg_goals, home_adv = _estimate_strengths(
        matches,
        smooth=smooth,
        avg_goals_baseline=avg_goals_baseline,
        home_adv_baseline=home_adv_baseline,
    )

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    relegated = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            strengths,
            avg_goals,
            home_adv,
            team_home_advantages,
            rng,
        )
        for team in table.tail(4)["team"]:
            relegated[team] += 1

    for t in relegated:
        relegated[t] /= iterations
    return relegated


def simulate_final_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    team_home_advantages: Dict[str, float] | None = None,
    smooth: float = 1.0,
    avg_goals_baseline: float = 2.5,
    home_adv_baseline: float = 1.0,
    home_smooth: float = 0.0,
    home_baseline: float | None = None,
) -> pd.DataFrame:
    """Project average finishing position and points."""
    if rng is None:
        rng = np.random.default_rng()

    team_home_advantages = _prepare_team_home_advantages(
        matches,
        team_home_advantages,
        smooth=home_smooth,
        baseline=home_baseline,
    )
    strengths, avg_goals, home_adv = _estimate_strengths(
        matches,
        smooth=smooth,
        avg_goals_baseline=avg_goals_baseline,
        home_adv_baseline=home_adv_baseline,
    )

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    pos_totals = {t: 0.0 for t in teams}
    points_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            strengths,
            avg_goals,
            home_adv,
            team_home_advantages,
            rng,
        )
        for idx, row in table.iterrows():
            pos_totals[row["team"]] += idx + 1
            points_totals[row["team"]] += row["points"]

    results = []
    for team in teams:
        results.append(
            {
                "team": team,
                "position": pos_totals[team] / iterations,
                "points": points_totals[team] / iterations,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("position").reset_index(drop=True)
    return df


def summary_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    *,
    rng: np.random.Generator | None = None,
    team_home_advantages: Dict[str, float] | None = None,
    smooth: float = 1.0,
    avg_goals_baseline: float = 2.5,
    home_adv_baseline: float = 1.0,
    home_smooth: float = 0.0,
    home_baseline: float | None = None,
) -> pd.DataFrame:
    """Return a combined projection table."""
    chances = simulate_chances(
        matches,
        iterations=iterations,
        rng=rng,
        team_home_advantages=team_home_advantages,
        smooth=smooth,
        avg_goals_baseline=avg_goals_baseline,
        home_adv_baseline=home_adv_baseline,
        home_smooth=home_smooth,
        home_baseline=home_baseline,
    )
    relegation = simulate_relegation_chances(
        matches,
        iterations=iterations,
        rng=rng,
        team_home_advantages=team_home_advantages,
        smooth=smooth,
        avg_goals_baseline=avg_goals_baseline,
        home_adv_baseline=home_adv_baseline,
        home_smooth=home_smooth,
        home_baseline=home_baseline,
    )
    table = simulate_final_table(
        matches,
        iterations=iterations,
        rng=rng,
        team_home_advantages=team_home_advantages,
        smooth=smooth,
        avg_goals_baseline=avg_goals_baseline,
        home_adv_baseline=home_adv_baseline,
        home_smooth=home_smooth,
        home_baseline=home_baseline,
    )

    table = table.sort_values("position").reset_index(drop=True)
    table["position"] = range(1, len(table) + 1)
    table["points"] = table["points"].round().astype(int)
    table["title"] = table["team"].map(chances)
    table["relegation"] = table["team"].map(relegation)
    return table[["position", "team", "points", "title", "relegation"]]
