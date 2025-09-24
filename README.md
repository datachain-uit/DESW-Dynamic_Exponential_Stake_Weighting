## Project Overview

This repository contains two complementary projects used to analyze and simulate decentralization in Proof-of-Stake (PoS) networks:

- `analysis_chains/`: Fetches validator data from multiple blockchains, computes decentralization metrics (Gini, Nakamoto, HHI) under multiple weighting schemes (w, SRSW, LOG, DESW), and exports tables and charts.
- `pos_simulator_python/`: Simulates various PoS algorithms (Weighted, SRSW, Log-weighted, DESW, etc.), tracks the evolution of Gini/HHI/Nakamoto across epochs, and compares methods.

---

## 1) analysis_chains

### What it does

- Collects validator and token data for: `aptos, axelar, celestia, celo, ethereum, injective, polygon, sui` (see `analysis_chains/chains/`).
- Stores daily CSVs in `analysis_chains/data/` using names like `{ddmmyyyy}_{chain}.csv`.
- Computes decentralization metrics in `analysis_chains/analysis/metrics.py`:
  - Standard: Gini, Nakamoto, HHI
  - Weighted variants: SRSW (sqrt), LOG (log1p), DESW (dynamic exponent derived from Gini)
- Produces comparison charts and a consolidated CSV in `analysis_chains/results/`.

### Setup

Requires Python 3.10+.

```bash
cd analysis_chains
pip install -r requirements.txt
```

If you fetch Ethereum data, set `DUNE_API_KEY` in your environment (loaded via `dotenv`).

PowerShell (Windows):

```powershell
$env:DUNE_API_KEY = "<YOUR_DUNE_API_KEY>"
```

Bash:

```bash
export DUNE_API_KEY="<YOUR_DUNE_API_KEY>"
```

### Fetch all chains for the current day

```bash
cd analysis_chains
python main.py
```

Note: If Ethereum is enabled, ensure `DUNE_API_KEY` is set before running.

### Fetch a single chain (optional)

```bash
python -m chains.aptos
python -m chains.axelar
python -m chains.celestia
python -m chains.celo
python -m chains.ethereum   # requires DUNE_API_KEY
python -m chains.injective
python -m chains.polygon
python -m chains.sui
```

These commands will write CSVs into `analysis_chains/data/` for the current date.

### Compute metrics and plot charts

Main script: `analysis_chains/analysis/metrics.py` (the `main()` function sets a default `date = '02092025'`; change this to match your collected CSVs).

```bash
cd analysis_chains/analysis
python metrics.py
```

Outputs (in `analysis_chains/results/`):

- Consolidated CSV: `{date}_blockchain_metrics.csv`
- Charts: `{date}_gini_index_comparison.png`, `{date}_hhi_coefficients_comparison.png`, `{date}_nakamoto_coefficients_combined.png`

Implementation note: `analysis_chains/analysis/coefficient.py` provides the reference formulas for Gini, Nakamoto, and HHI and is reused for weighted variants.

---

## 2) pos_simulator_python

### What it does

- Simulates multiple PoS consensus variants with different initial stake distributions and join/leave events:
  - `WEIGHTED`, `SRSW_WEIGHTED`, `LOG_WEIGHTED`, `DESW`, `OPPOSITE_WEIGHTED`, `GINI_STABILIZED`, `RANDOM` (see `src/parameters.py` and `src/utils.py`).
- Tracks Gini, HHI, Nakamoto, and peer count history across epochs.
- Saves results and plots under `experiments/*/results/` and `experiments/results/`.

### Origin and differences from the original project

- This simulator is a Python re-implementation and extension of the original Julia project: [`lorenzorovida/PoS-Simulator`](https://github.com/lorenzorovida/PoS-Simulator).
- Key additions/differences in this Python version:
  - Adds the Nakamoto coefficient alongside Gini and HHI.
  - Adds and compares SRSW (Square-Root of Stake Weight), LSW/LOG (Log-weighted), and DESW (Dynamic Exponential Stake Weighting) algorithms.
  - Adds Scheduled Joins (predefined epochs and stake amounts for new peers).
  - Bundles comparison/benchmark scripts, exporting plots and JSON summaries.

### Setup

```bash
cd pos_simulator_python
pip install -r requirements.txt
```

### Quick start: run experiments

```bash
# Aggregate comparison across algorithms
python experiments/comparison.py

# Individual algorithms
python experiments/weighted_pos/weighted_experiment.py
python experiments/srsw_weighted_pos/srsw_weighted_experiment.py
python experiments/log_weighted_pos/log_weighted_experiment.py
python experiments/desw_pos/desw_experiment.py
python experiments/opposite_weighted_pos/opposite_weighted_experiment.py
python experiments/gini_stabilized_pos/gini_stabilized_experiment.py
```

Examples of outputs:

- `experiments/results/gini_comparison.png`, `experiments/results/nakamoto_comparison.png`
- `experiments/desw_pos/results/*` (Gini/HHI/Nakamoto/Peers over time)

### Benchmark (primary experiments)

The benchmark aggregates the core scenarios and exports JSON summaries:

```bash
cd pos_simulator_python
python benchmark/benchmark_algorithms.py
```

Results are saved under: `pos_simulator_python/benchmark/results/` (e.g., `benchmark_summary_scenario_1.json`, `benchmark_summary_scenario_2.json`, `benchmark_summary_scenario_3.json`).

### Customize simulation parameters

Parameters are defined in `src/parameters.py` (`dataclass Parameters`), e.g.:

- `n_epochs`, `n_peers`, `initial_gini`, `p_join`, `p_leave`, `penalty_percentage`, `reward`, `scheduled_joins`, etc.
- Enums `PoS`, `Distribution`, `NewEntry`, `SType` select algorithm/distribution/join behavior/Gini-stabilization model.

You can edit the experiment files in `experiments/*_experiment.py` or import from `src.simulator` to build your own scripts.

## Folder structure (short)

```text
analysis_chains/
  chains/                 # Per-chain data collectors
  analysis/
    coefficient.py        # Gini/Nakamoto/HHI formulas
    metrics.py            # Aggregation and plotting
  data/                   # Daily CSVs
  results/                # Consolidated CSVs and charts

pos_simulator_python/
  src/                    # Core algorithms and simulation
  experiments/            # Experiments + outputs
  benchmark/              # Benchmarks across scenarios
```

---

## System requirements

- Python 3.10+
- Internet access for public APIs when collecting chain data (especially `analysis_chains/chains/*`).
- `DUNE_API_KEY` env var for Ethereum data collection.

---

## Troubleshooting

- Empty files or missing `tokens` column: such files are skipped; re-check the upstream API and try again later.
- No charts saved: verify write permissions for `results/` or headless Matplotlib configuration (we use `savefig`).
- Ethereum fetch failures: ensure `DUNE_API_KEY` is set and the quota is sufficient.

---

## Licenses

- `analysis_chains` includes an MIT License (`analysis_chains/LICENSE`).
- `pos_simulator_python` includes its own MIT License (`pos_simulator_python/LICENSE`).

Both are permissive: you may use/modify/distribute, provided you keep copyright and license notices.

---

## Acknowledgements

- Original Julia simulator: [`lorenzorovida/PoS-Simulator`](https://github.com/lorenzorovida/PoS-Simulator)
