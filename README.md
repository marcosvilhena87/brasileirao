# Brasileirão Simulator

This project provides a simple simulator for the 2025 Brasileirão Série A season. It parses the fixtures provided in `data/Brasileirao2025A.txt`, builds a league table from played matches and simulates the remaining games many times to estimate title and relegation probabilities.

## Usage

Install dependencies from `requirements.txt` and run the simulator:

```bash
pip install -r requirements.txt
python main.py --simulations 1000
```

You can customise the strength estimation with additional options. For example:

```bash
python main.py --simulations 1000 --smooth 2 \
    --home-smooth 1 --avg-goals-baseline 2.6 --home-adv-baseline 1.3
```

The default home advantage baseline is ``1.3``, roughly matching the
average home advantage observed in recent Série A seasons.

### Default parameters

The command-line tool exposes several options for tweaking the model. Their
default values follow the SportsClubStats approach:

- `--smooth`: ``1.0``
- `--avg-goals-baseline`: ``2.5``
- `--home-adv-baseline`: ``1.3``
- `--home-smooth`: ``0.0``
- `--home-baseline`: ``None``

By default the simulator uses a SportsClubStats-style rating model. Team attack and defence strengths are based on goals scored and conceded so far in the season. Remaining fixtures are simulated with Poisson-distributed scores using these strengths.

The script outputs the estimated chance of winning the title for each team. It then prints the probability of each side finishing in the bottom four and being relegated. It also estimates the average final position and points of every club.

## Tie-break Rules

When building the league table teams are ordered using the official Série A criteria:

1. Points
2. Number of wins
3. Goal difference
4. Goals scored
5. Points obtained in the games between the tied sides
6. Team name (alphabetical)

These rules are implemented in :func:`league_table` and therefore affect all simulation utilities.

## Project Layout

- `data/` – raw fixtures and results.
- `src/brasileirao/simulator.py` – parsing, table calculation and simulation routines.
- `main.py` – command-line interface to run the simulation.
- `tests/` – basic unit tests.

The main functions can be imported directly from the package:

```python
from brasileirao import (
    parse_matches,
    league_table,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
)
```

## License

This project is licensed under the [MIT License](LICENSE).
