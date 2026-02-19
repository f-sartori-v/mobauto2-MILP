# mobauto2-benders
Benders decomposition approach for the MobAuto2 project.

Quick start (CLI)
- From repo root after installing the package:
- `pip install -e .`
- `mobauto2-benders run`
- `python -m mobauto2_benders run`

Quick start (Python)
```python
from mobauto2_benders.app import run

result = run()  # uses configs/default.yaml
print(result.status, result.iterations, result.best_lower_bound, result.best_upper_bound)
```

Other CLI commands
- `python -m mobauto2_benders info` prints key config settings.
- `python -m mobauto2_benders validate` checks config and problem stubs.

Multi-resolution run (coarse -> fine)
- `python -m mobauto2_benders run --multi-res 30,15,5,1`

Configuration (v2)
- Default config: `configs/default.yaml`.
- Schema header:
- `schema.name` must be `mobauto2_benders_config`
- `schema.version` must be `2`
- Sections:
- `run`: logging/report paths, run name, optional seed
- `data`: demand source, scenario files and weights
- `model`: time discretization, fleet, energy, and cost parameters
- `master`: master-structure toggles
- `subproblem`: cut generation and subproblem structure
- `solver`: Benders loop and backend solver settings

Example `configs/default.yaml` (commented)
```yaml
schema:
  name: mobauto2_benders_config
  version: 2

run:
  # Optional run label for logs/outputs
  name: default
  # log_level controls verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_level: INFO
  # Optional paths for log/report outputs
  log_file: null
  report_dir: null
  # Optional random seed (set to null to disable)
  seed: 42

data:
  # Demand source (preferred): file with requests or R_out/R_ret arrays
  demand_file: setups/random.yaml
  # Optional scenario files for multi-scenario runs
  scenario_files: []
  # Optional weights aligned with scenario_files
  scenario_weights: null
  # Optional inline demand arrays (single scenario)
  R_out: null
  R_ret: null

model:
  time:
    # Total horizon length in minutes (T_minutes) or discrete slots (T)
    T_minutes: 810
    # Minutes per slot
    slot_resolution: 30
    # Trip duration in minutes (converted internally to slots)
    trip_duration_minutes: 30
  fleet:
    # Number of shuttles
    Q: 2
    # Initial battery per shuttle (length Q)
    binit: [150.0, 150.0]
  energy:
    # Battery capacity
    Emax: 150
    # Energy consumed when starting a trip
    L: 30
    # Max charge added per slot when charging; expressions allowed here only
    delta_chg: 70 / (60 / slot_resolution)
  costs:
    # Small penalty on starting trips
    start_cost_epsilon: 0.01
    # Small penalty per extra concurrent departure beyond 1 per slot
    concurrency_penalty: 0.25

master:
  # Optional FIFO symmetry-breaking across vehicles
  use_fifo_symmetry: true
  # Symmetry breaking: order vehicles by total departures
  symmetry_breaking: true
  # Stronger aggregated cuts by tau
  aggregate_cuts_by_tau: true
  # Use one theta per scenario (requires multi-cuts)
  theta_per_scenario: true
  # Drop tiny cut coefficients to improve numerics
  cut_coeff_threshold: 0.001
  # Debug: write LP after each new cut
  write_lp_after_cut: false

subproblem:
  # Multi-cuts vs averaged cut across scenarios
  multi_cuts_by_scenario: true
  # Magnanti–Wong selection for Pareto-optimal cuts
  use_magnanti_wong: true
  # Core-point mixing factor alpha in (0,1]
  mw_core_alpha: 0.3
  # Use dual slopes for cut generation
  use_dual_slopes: false
  # Capacity units contributed by one vehicle starting at time tau
  S: 15
  # Max waiting time in minutes or slots
  Wmax_minutes: 30
  # Penalty per unit of unserved demand
  p: 50.0
  # Tiny tie-breaker cost encouraging packing
  fill_first_epsilon: 1.0e-6
  # Penalty per unit of unused seat-capacity
  unused_capacity_penalty: 0.5

solver:
  # Benders loop settings
  max_iterations: 100
  tolerance: 0.001
  time_limit_s: 600
  # Optional stall-based early stopping
  stall_max_no_improve_iters: 0
  stall_min_abs_improve: 1.0
  stall_min_rel_improve: 0.002
  # Backend solver names (Pyomo)
  master_solver: cplex
  subproblem_solver: cplex_direct
  # Emit solver logs during MP solves
  solver_tee: false
```

Requirements (runtime)
- Python 3.10 venv with `pyomo` available (models are built with Pyomo).
- A MILP/LP solver supported by Pyomo. Default config uses CPLEX:
- Master: `solver.master_solver` (default `cplex`)
- Subproblem: `solver.subproblem_solver` (default `cplex_direct`)
- `pyyaml` is required if you load YAML configs or demand files.

Troubleshooting
- Import errors: add `src` to `PYTHONPATH`, or run `python -m mobauto2_benders`.
- PyYAML error: install `pyyaml` to read YAML configs/demand files.
- CPLEX bindings error: make sure you are in the Python 3.10 venv where CPLEX is installed.
