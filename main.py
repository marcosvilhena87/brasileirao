import argparse
from src.brasileirao.simulator import parse_matches, simulate_chances, league_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Brasileirão 2025 title odds")
    parser.add_argument("--file", default="data/Brasileirao2025A.txt", help="fixture file path")
    parser.add_argument("--simulations", type=int, default=1000, help="number of simulation runs")
    parser.add_argument(
        "--rating",
        default="ratio",
        choices=["ratio", "poisson"],
        help="team strength estimation method",
    )
    args = parser.parse_args()

    matches = parse_matches(args.file)
    chances = simulate_chances(
        matches, iterations=args.simulations, rating_method=args.rating
    )

    print("Title chances:")
    for team, prob in sorted(chances.items(), key=lambda x: x[1], reverse=True):
        print(f"{team:15s} {prob:.2%}")


if __name__ == "__main__":
    main()
