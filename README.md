# mobauto2-benders
Benders decomposition approach for the MobAuto2 project.

Quick start (CLI)
- From repo root after installing the package:
  - `pip install -e .`
  - `python -m mobauto2_benders run`
  - `python -m mobauto2_benders run --config configs/default.yaml`
- As a script (no `PYTHONPATH` needed):
  - `python src/mobauto2_benders/cli.py run`

Quick start (Python)
```
from mobauto2_benders import run

result = run()  # uses configs/default.yaml
print(result.status, result.iterations, result.best_lower_bound, result.best_upper_bound)
```

Other CLI commands
- `python -m mobauto2_benders info` prints the resolved config.
- `python -m mobauto2_benders validate` checks config and problem stubs.

Multi-resolution run (coarse -> fine)
- `python -m mobauto2_benders run --multi-res 30,15,5,1`

Configuration
- Default config: `configs/default.yaml`.
- Demand input: `setups/demo_cont_demand.yaml` is referenced by default.
- Edit the YAML to enable features like Magnanti-Wong cuts:
  - `subproblem.params.use_magnanti_wong: true`
- Scenario runs: set `subproblem.params.scenarios` or `subproblem.params.scenario_files`.

Requirements (runtime)
- Python 3.10 venv with `pyomo` available (models are built with Pyomo).
- A MILP/LP solver supported by Pyomo. Default config uses CPLEX:
  - Master: `master.params.solver` (default `cplex_persistent`)
  - Subproblem: `subproblem.params.lp_solver` (default `cplex_direct`)
- `pyyaml` is required if you load YAML configs or demand files.

Troubleshooting
- Import errors: add `src` to `PYTHONPATH`, or run `cli.py` directly.
- PyYAML error: install `pyyaml` to read YAML configs/demand files.
- CPLEX bindings error: make sure you are in the Python 3.10 venv where CPLEX is installed.
