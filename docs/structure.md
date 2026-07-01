# Repository layout

```
text2reward-Rag/
├── train.py / evaluate.py / plot_training.py
├── reward_program.py
├── txt2reward/
│   ├── core/           # metrics, logging, types
│   ├── config/         # env, paths, training, llm, fitness, validation
│   ├── archive/        # store, fitness, curriculum, retrieval
│   ├── llm/            # designer, prompts, validation, key_manager
│   ├── sandbox/        # AST validation + safe execution
│   ├── trajectory/     # trajectory bank (smoke-test Stage B)
│   ├── reward/         # LLMRewardWrapper
│   ├── training/       # train, env_factory, callbacks, logger, plots
│   └── evaluation/
├── scripts/            # see scripts/README.md
├── examples/
├── tests/
└── docs/
```

| Module | Responsibility |
|--------|----------------|
| `archive/store.py` | persistence, CRUD, RAG formatting |
| `archive/fitness.py` | fitness v6/v7/v8 |
| `llm/designer.py` | evolution orchestration |
| `llm/validation.py` | smoke-test + Stage B gates |
| `training/train.py` | CLI → PPO + callbacks |
| `sandbox/sandbox.py` | AST validation, timed execution |
| `reward/wrapper.py` | observation parsing, shaped reward, episode stats |

Generated reward code is AST-validated, executed in a restricted namespace with
timeouts, and re-validated on archive restore.

Runtime: `requirements.txt`. Tests: `requirements-dev.txt` (`pytest`).

---

## Safety, clipping, and validation gates

These fixes address the crash-farming / evolution-freeze issues identified in the
technical review. See `docs/baselines/` for measured thresholds.

### Per-step reward clipping

| Constant | Default | Applies when |
|----------|---------|--------------|
| `REWARD_STEP_CLIP_MIN/MAX` | `-10` / `+10` | Normal steps |
| `REWARD_COLLISION_CLIP_MIN/MAX` | `-120` / `0` | `collided=True` |

Implementation: `txt2reward/reward/clip.py` (`clip_shaped_reward`). The wrapper
and all validation gates call the same function so PPO sees the same signal the
smoke tests score.

**Debug:** set `DEBUG_REWARD=1` during training. The wrapper logs raw vs clipped
reward on every collision step and every 1000th step (see `LLMRewardWrapper.step`).

### Validation pipeline parity

`txt2reward/llm/validation.py` runs Stage A (fast smoke) then Stage B (trajectory
bank). Stage B uses `_runtime_step_reward()` → `clip_reward_for_state()` so
unclipped rewards cannot pass while clipped training would crash-farm.

Stage B bank size: `TRAJECTORY_BANK_MODE=lite|full` (default **lite**, ~16 trajectories).
Curriculum phase adjusts soft-rate ceiling, passive tolerance, and (on lite banks)
an absolute soft-violation cap — see `txt2reward/config/validation.py`.

### Evolution freeze escape

When `crash_rate ≥ evolve-max-crash-rate` (default 70%) for `--max-freeze-windows`
consecutive windows (default 3), `RewardDesigner` forces one archive + LLM attempt
instead of deadlocking. Config: `train.py --max-freeze-windows`.

### Gate telemetry

`record_smoke_gate_failure()` / `smoke_gate_failure_counts()` track which Stage A/B
check rejected candidates. Printed by `scripts/calibrate_smoke_gate.py` and logged
when evolution exhausts retries (`RewardDesigner`).

### Calibration

```bash
python scripts/calibrate_smoke_gate.py
```

Writes `docs/baselines/phase4-post-clip-gates.json` with bootstrap soft/passive
rates, phase thresholds, and accumulated gate failure counts.
