# Brasileirão Simulator

This project provides a simple simulator for the 2024 Brasileirão Série A season. It parses the fixtures provided in `data/Brasileirao2024A.txt`, builds a league table from played matches, and simulates the remaining games many times to estimate title probabilities.

## Usage

Install dependencies and run the simulator:

```bash
pip install pandas numpy
python main.py --simulations 1000
```

The script outputs the estimated chance of winning the title for each team.

## Project Layout

- `data/` – raw fixtures and results.
- `src/brasileirao/simulator.py` – parsing, table calculation and simulation routines.
- `main.py` – command-line interface to run the simulation.
- `tests/` – basic unit tests.
