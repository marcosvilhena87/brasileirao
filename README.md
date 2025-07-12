# Brasileirão Simulator

This project provides a simple simulator for the 2025 Brasileirão Série A season. It parses the fixtures provided in `data/Brasileirao2025A.txt`, builds a league table from played matches, and simulates the remaining games many times to estimate title probabilities.

## Usage

Install dependencies and run the simulator:

```bash
pip install pandas numpy statsmodels
python main.py --simulations 1000 --rating poisson
```

The `--rating` option accepts `ratio` (default), `historic_ratio`, `poisson`, or
`elo` to choose how team strengths are estimated. The `historic_ratio` method
mixes results from the 2024 season with a lower weight. The `elo` method
updates team ratings over time using an Elo formula; the `simulate_chances`
function exposes an `elo_k` parameter for deterministic runs. Use the
`--seed` option to set a random seed and reproduce a specific simulation.

The script outputs the estimated chance of winning the title for each team.

## Project Layout

- `data/` – raw fixtures and results.
- `src/brasileirao/simulator.py` – parsing, table calculation and simulation routines.
- `main.py` – command-line interface to run the simulation.
- `tests/` – basic unit tests.

The main functions can be imported directly from the package:

```python
from brasileirao import parse_matches, league_table, simulate_chances
```
