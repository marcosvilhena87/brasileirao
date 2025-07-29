"""Command-line interface for running Brasileir\u00e3o simulations."""

# pylint: disable=wrong-import-position

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import numpy as np
from brasileirao import (
    parse_matches,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
    summary_table,
    league_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Brasileirão 2025 title odds")
    parser.add_argument("--file", default="data/Brasileirao2025A.txt", help="fixture file path")
    parser.add_argument("--simulations", type=int, default=1000, help="number of simulation runs")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for repeatable simulations",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=1.0,
        help="strength smoothing factor",
    )
    parser.add_argument(
        "--avg-goals-baseline",
        type=float,
        default=2.5,
        help="baseline average goals when no data",
    )
    parser.add_argument(
        "--home-adv-baseline",
        type=float,
        default=1.0,
        help="baseline league home advantage",
    )
    parser.add_argument(
        "--home-smooth",
        type=float,
        default=0.0,
        help="smoothing factor for team home advantage",
    )
    parser.add_argument(
        "--home-baseline",
        type=float,
        default=None,
        help="override baseline for team home advantage",
    )
    args = parser.parse_args()

    matches = parse_matches(args.file)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    chances = simulate_chances(
        matches,
        iterations=args.simulations,
        rng=rng,
        smooth=args.smooth,
        avg_goals_baseline=args.avg_goals_baseline,
        home_adv_baseline=args.home_adv_baseline,
        home_smooth=args.home_smooth,
        home_baseline=args.home_baseline,
    )
    relegation = simulate_relegation_chances(
        matches,
        iterations=args.simulations,
        rng=rng,
        smooth=args.smooth,
        avg_goals_baseline=args.avg_goals_baseline,
        home_adv_baseline=args.home_adv_baseline,
        home_smooth=args.home_smooth,
        home_baseline=args.home_baseline,
    )
    table_proj = simulate_final_table(
        matches,
        iterations=args.simulations,
        rng=rng,
        smooth=args.smooth,
        avg_goals_baseline=args.avg_goals_baseline,
        home_adv_baseline=args.home_adv_baseline,
        home_smooth=args.home_smooth,
        home_baseline=args.home_baseline,
    )

    summary = table_proj.copy()
    summary["title"] = summary["team"].map(chances)
    summary["relegation"] = summary["team"].map(relegation)
    summary = summary.sort_values("position").reset_index(drop=True)
    summary["position"] = range(1, len(summary) + 1)
    summary["points"] = summary["points"].round().astype(int)

    TITLE_W = 7
    REL_W = 10
    print(f"{'Pos':>3}  {'Team':15s} {'Points':>6} {'Title':^{TITLE_W}} {'Relegation':^{REL_W}}")
    for _, row in summary.iterrows():
        title = f"{row['title']:.2%}"
        releg = f"{row['relegation']:.2%}"
        print(
            f"{row['position']:>2d}   {row['team']:15s} {row['points']:6d} {title:^{TITLE_W}} {releg:^{REL_W}}"
        )


if __name__ == "__main__":
    main()
