# Scripts

Maintenance utilities that are not part of the training runtime.

| Script | Purpose |
|--------|---------|
| `calibrate_smoke_gate.py` | Offline calibration of Stage B gates. Runs candidate rewards through `measure_gate_stats` and `_full_validation_pipeline`, prints `smoke_gate_failure_counts`, and writes `docs/baselines/phase4-post-clip-gates.json`. |

Run from the repository root:

```bash
python scripts/calibrate_smoke_gate.py
```
