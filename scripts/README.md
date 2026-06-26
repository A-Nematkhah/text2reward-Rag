# Scripts

Maintenance utilities that are not part of the training runtime.

| Script | Purpose |
|--------|---------|
| `calibrate_smoke_gate.py` | Offline calibration of Stage B trajectory-bank gate thresholds. Runs candidate reward bodies through `measure_gate_stats` and `_full_validation_pipeline` without calling the Groq API. |

Run from the repository root:

```bash
python scripts/calibrate_smoke_gate.py
```
