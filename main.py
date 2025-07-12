import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from brasileirao import parse_matches, simulate_chances, league_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Brasileirão 2025 title odds")
    parser.add_argument("--file", default="data/Brasileirao2025A.txt", help="fixture file path")
    parser.add_argument("--simulations", type=int, default=1000, help="number of simulation runs")
    parser.add_argument(
        "--rating",
        default="ratio",
        choices=[
            "ratio",
            "historic_ratio",
            "poisson",
            "neg_binom",
            "elo",
            "leader_history",
        ],
        help="team strength estimation method (use 'historic_ratio' to include past season)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for repeatable simulations",
    )
    parser.add_argument(
        "--elo-k",
        type=float,
        default=20.0,
        help="Elo K factor when using the 'elo' rating method",
    )
    parser.add_argument(
        "--leader-history-paths",
        nargs="*",
        default=["data/Brasileirao2024A.txt"],
        help="Past season files for leader_history rating method",
    )
    parser.add_argument(
        "--leader-weight",
        type=float,
        default=0.5,
        help="Weight for leader_history influence",
    )
    args = parser.parse_args()

    matches = parse_matches(args.file)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    chances = simulate_chances(
        matches,
        iterations=args.simulations,
        rating_method=args.rating,
        rng=rng,
        elo_k=args.elo_k,
        leader_history_paths=args.leader_history_paths,
        leader_history_weight=args.leader_weight,
    )

    print("Title chances:")
    for team, prob in sorted(chances.items(), key=lambda x: x[1], reverse=True):
        print(f"{team:15s} {prob:.2%}")


if __name__ == "__main__":
    main()
