# txt2reward-v2

**A Text-to-Reward system for highway-driving reinforcement learning.**

An LLM (Groq / Llama 3.3 70B) writes complete Python reward functions for a PPO
agent in `highway-env`, trains the agent, evaluates the resulting behaviour,
critiques itself for reward hacking, and evolves an improved reward program —
an evolutionary search over reward *code*, not just reward *weights*.

```
"Drive fast, overtake safely, avoid collisions"
                    │
                    ▼
            ┌───────────────┐
            │ RewardDesigner│  Groq API (Llama 3.3 70B)
            │  + RAG context│  ← reward_archive.json (top performers)
            └───────┬───────┘
                    │  generates compute_reward(state) source
                    ▼
          ┌─────────────────────┐
          │   Sandbox Pipeline   │  AST validation → smoke test →
          │                     │  trajectory safety gate
          └──────────┬──────────┘
                    │ pass
                    ▼
            reward_program.py ◄────────────────┐
                    │   (hot-reloaded by workers)│
                    ▼                           │
              PPO Training (SB3)                │
                    │                           │
                    ▼                           │
           evaluate_agent() → metrics           │
                    │                           │
                    ▼                           │
            LLM Critique (hacking detection)    │
                    │                           │
                    ▼                           │
        reward_archive.json (code + metrics)    │
                    │                           │
                    └── next generation ─────────┘
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
  - [Google Colab](#google-colab)
  - [Local setup](#local-setup)
- [CLI reference](#cli-reference)
- [How it works](#how-it-works)
  - [State object](#state-object)
  - [Reward program sandbox](#reward-program-sandbox)
  - [Evolutionary loop](#evolutionary-loop)
  - [Fitness function](#fitness-function)
  - [Reward-hacking detection](#reward-hacking-detection)
  - [Multi-process architecture](#multi-process-architecture)
- [Output files](#output-files)
- [Migrating from the weight-tuning version](#migrating-from-the-weight-tuning-version)
- [Known limitations](#known-limitations)
- [License](#license)
- [References](#references)

---

## Why this exists

Earlier versions of this project had the LLM tune a fixed set of scalar
weights (`w_speed`, `w_safety`, …) plugged into a hand-written reward
formula. That approach caps the agent's ceiling at whatever behaviours the
human-designed formula can express — the LLM can only turn knobs, not change
the shape of the function.

**txt2reward-v2 removes that ceiling.** The LLM writes the reward formula
itself — a complete, sandboxed Python function operating on a structured
state object — and iterates on the *code*, not the coefficients.

---

## Architecture

| Stage | Component | Responsibility |
|---|---|---|
| 1 | `txt2reward.llm.RewardDesigner` | Sends the driving goal + archive context to Groq, receives generated reward code |
| 2 | `txt2reward.sandbox` | AST-validates the code (no imports/exec/eval/loops/attribute access) |
| 3 | `txt2reward.llm.validation` | Smoke tests and trajectory-bank safety gate before code reaches disk |
| 4 | `reward_program.py` | The current reward function, hot-swapped on disk |
| 5 | `txt2reward.reward.LLMRewardWrapper` | Per-worker Gym wrapper; reloads and executes `reward_program.py` |
| 6 | `txt2reward.evaluation.evaluate_agent()` | Measures speed, crash rate, overtakes, completion rate |
| 7 | `txt2reward.archive` | Computes fitness, stores every generation, serves RAG-style context for the next generation |
| 8 | LLM critique | Flags reward-hacking patterns and proposes concrete fixes |

---

## Project structure

```
text2reward-Rag/
├── train.py                 # CLI entry — PPO training + evolution
├── evaluate.py              # CLI entry — evaluate trained models
├── plot_training.py         # CLI entry — training / evolution charts
├── reward_program.py        # Active reward function (hot-reloaded at runtime)
├── txt2reward/              # Main Python package
│   ├── config/              # Paths, PPO schedule, LLM, fitness, validation gates
│   ├── core/                # Metrics, logging, shared types
│   ├── archive/             # Fitness, archive persistence, RAG retrieval
│   ├── llm/                 # RewardDesigner, prompts, validation, Groq key rotation
│   ├── sandbox/             # AST validation + restricted execution
│   ├── trajectory/          # Synthetic trajectory bank (smoke-test Stage B)
│   ├── reward/              # LLMRewardWrapper (+ legacy components)
│   ├── training/            # Training loop, logger, plots
│   └── evaluation/          # Model evaluation pipeline
├── scripts/                 # Maintenance utilities (see scripts/README.md)
├── examples/                # Usage pointers (CLI workflows)
├── tests/                   # Pytest suite (115 tests)
├── docs/                    # Layout and design notes
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # pytest (includes -r requirements.txt)
└── LICENSE                  # MIT
```

Legacy flat imports (`reward_archive`, `reward_designer`, …) were removed in v2 layout;
use `txt2reward.<subpackage>` instead (e.g. `from txt2reward.archive import RewardArchive`).

---

## Requirements

- Python 3.10+
- A [Groq API key](https://console.groq.com) — the free tier is sufficient
- A GPU is recommended (Colab T4 works fine; CPU training is slow)

```bash
pip install -r requirements.txt

# Development / tests
pip install -r requirements-dev.txt
```

---

## Quickstart

### Google Colab

Use a GPU runtime (**Runtime → Change runtime type → T4 GPU**), clone this repo,
install dependencies, and run the same CLI commands as local setup. Set
`GROQ_API_KEY` in the environment (or use `api_keys.json` at the repo root).
Use `--drive-dir` on `train.py` to sync checkpoints and logs to Google Drive.

See `examples/README.md` for command templates.

### Local setup

```bash
export GROQ_API_KEY="gsk_xxxxxxxx"

# Train with a natural-language goal (auto-bootstraps reward_program.py)
python train.py --timesteps 200000 --n-envs 4 \
  --goal "Drive fast and safely, overtake slow vehicles, avoid collisions, minimise harsh braking."

# Evaluate the trained model with the current reward program
python evaluate.py --model ppo_highway_txt2reward.zip --episodes 10

# Evaluate against a specific earlier generation from the archive
python evaluate.py --model ppo_highway_txt2reward.zip --generation 2
```

> **Multiple Groq keys?** Copy `api_keys.json.example` to `api_keys.json` at the
> repo root (see `txt2reward.llm.key_manager`) and the designer will
> automatically rotate to the next available key on rate limits instead of
> stalling the run.

---

## CLI reference

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--timesteps` | `200000` | Total environment steps |
| `--n-envs` | `4` | Number of parallel environments |
| `--reload-interval` | `200` | Steps between `reward_program.py` reloads in each worker |
| `--evolve-every` | `100` | Generate a new reward program every N episodes (after warmup) |
| `--warmup-episodes` | `80` | Episodes before the first LLM reward generation |
| `--goal` | *(driving goal)* | Natural-language goal sent to the LLM |
| `--reward-path` | `reward_program.py` | Output path for the generated reward program |
| `--archive-file` | `reward_archive.json` | Path to the reward archive |
| `--bootstrap` | off | Force-generate an initial reward program before training |
| `--resume` | `None` | Checkpoint `.zip` to resume from |
| `--fresh` | off | Wipe log, archive, reward program, and checkpoints before starting |
| `--checkpoint-freq` | `10000` | Steps between checkpoints |
| `--drive-dir` | `/content/drive/MyDrive/txt2reward` | Google Drive folder for Colab sync |

### `evaluate.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `ppo_highway_txt2reward.zip` | Path to the trained model |
| `--episodes` | `10` | Number of evaluation episodes |
| `--no-shaped` | off | Disable shaped reward (env reward only) |
| `--reward-path` | `reward_program.py` | Reward program to evaluate with |
| `--generation` | `None` | Evaluate using a specific archived generation |
| `--render` | off | Render the environment visually |
| `--stochastic` | off | Use a stochastic policy (default: deterministic) |
| `--save` | `None` | Save results as JSON to this path |

---

## How it works

### State object

On every step, the environment observation is parsed into a structured state
dict and passed to the generated reward function:

```python
state = {
    "speed_ms":        25.3,   # ego speed [m/s]
    "front_dist":      48.0,   # distance to front vehicle [m]
    "ttc":             9.6,    # time-to-collision [s], capped at 30
    "rel_vel_ms":      -2.1,   # v_front - v_ego [m/s]
    "lane":            1,      # lane index, 0 = rightmost
    "overtook":        False,  # completed an overtake this step
    "lane_changed":    False,  # lane changed since last step
    "collided":        False,  # collision this step
    "nearby_vehicles": 3,      # vehicles within ~30 m
    "accel_ms2":       0.4,    # longitudinal acceleration [m/s²]
    "long_jerk":       0.1,    # longitudinal jerk [m/s³]
    "lat_jerk":        0.0,    # lateral jerk [m/s³]
}
```

### Reward program sandbox

The LLM generates a complete `compute_reward(state) -> float` function (see
`reward_program.py` for the current example). Every generated function is
sandboxed and validated before it can run:

- **No** `import`, `eval`/`exec`, file or network access, attribute access,
  or loops.
- Only an approved set of math functions (`sqrt`, `exp`, `log`, trig, `clip`,
  …) and approved state keys.
- **AST validation** (`txt2reward.sandbox.validate_reward_code`) rejects
  structurally unsafe or malformed code before it is ever executed.
- **Smoke testing** (`txt2reward.llm.validation`) executes the function against
  representative sample states — catching runtime errors structural checks
  can't, such as a `KeyError` from `state["overtake"]` instead of the correct
  `state["overtook"]`.
- A **trajectory safety gate** simulates a cautious 40-step rollout and a
  reckless, crash-ending 40-step rollout, and rejects any reward function
  under which the reckless trajectory scores *higher* — a direct defence
  against reward hacking before training even starts.

Only code that survives all three checks is written to disk.

### Evolutionary loop

1. **Warmup** — the first N episodes train with the current/default reward
   program.
2. **Every `--evolve-every` episodes** after warmup:
   - Episode statistics are aggregated into evaluation metrics.
   - The reward program that just produced those metrics is **archived
     unconditionally** — the generation counter is *always* derived from
     `len(archive.entries)`, never tracked independently.
   - The LLM **critiques** that entry (reward-hacking detection, failure
     modes, proposed fixes), including an explicit comparison against the
     previous generation's trend.
   - The LLM **generates an improved reward program**, given the
     top-performing archive entries as RAG-style context.
   - The new program runs through the full sandbox pipeline; only on success
     does it replace `reward_program.py` and advance the generation. On any
     failure, the previous program stays in effect.
3. Worker environments reload `reward_program.py` from disk every
   `--reload-interval` steps.

### Fitness function

```
fitness = ( w_speed · speed_score + w_overtake · overtake_score
          + w_comfort · comfort_score + w_ttc · ttc_score
          + w_complete · completion_score ) × safety_gate(crash_rate)
```

Each component is normalised to `[0, 1]` independently, and a two-stage
multiplicative safety gate suppresses fitness sharply once `crash_rate`
exceeds 30%, with an additional hard penalty above 80% — ensuring a
crash-prone agent can never out-score a slower, safer one regardless of raw
speed or overtake count. See `txt2reward.archive.fitness` for the full
derivation and worked examples.

### Reward-hacking detection

The critique prompt explicitly looks for:

| Pattern | Signal |
|---|---|
| Oscillatory lane changes | many lane changes, few completed overtakes |
| Acceleration spam | high jerk/accel with no corresponding speed gain |
| Stationary reward farming | low speed paired with high shaped reward |
| TTC exploitation | very low time-to-collision without crashes (tailgating for a bonus) |
| "Slow down to survive" | speed and/or overtakes *decreasing* while crash rate improves — the agent learning passivity rather than skill |

The last pattern is detected automatically: every critique includes a
generation-over-generation trend comparison, and a hard-coded warning fires
whenever speed or overtakes drop while crash rate improves, so this failure
mode can't hide in metrics that look fine in isolation.

### Multi-process architecture

Training uses `SubprocVecEnv` — each worker runs in its own process. The
shared state between processes is `reward_program.py` on disk:

- **Main process** — runs `RewardDesigner`, calls Groq, validates and writes
  the new reward program, maintains `reward_archive.json`.
- **Worker processes** — run `LLMRewardWrapper`, periodically reloading and
  executing the reward program from disk.

---

## Output files

| File | Description |
|---|---|
| `ppo_highway_txt2reward.zip` | Final trained PPO model |
| `ppo_highway_*.zip` | Intermediate checkpoints |
| `reward_program.py` | Current generated reward function |
| `reward_archive.json` | Every generation: code, metrics, fitness, critique |
| `training_log.json` | Per-episode and per-generation training history |
| `tb_logs/` | TensorBoard training logs |

---

## Migrating from the weight-tuning version

If you used the earlier (weight-tuning) version of this project:

1. `reward_weights.json` is no longer the primary mechanism — it's replaced
   by `reward_program.py`. `txt2reward.reward.components` is kept only as a legacy
   compatibility layer for `evaluate.py --no-shaped`.
2. `RewardDesigner` no longer exposes `get_weights()` returning a weight
   dict; it now returns `{"generation": int, "reward_path": str}` for
   logging compatibility.
3. Existing `training_log.json` files from the old format are not compatible
   with the new per-episode schema (`generation` replaces weight snapshots)
   — start a fresh log or migrate manually.
4. Re-run with `--bootstrap` on first launch to generate an initial
   LLM-written reward program instead of relying on the bundled default in
   `reward_program.py`.

---

## Known limitations

- The Groq free tier is rate-limited. The designer backs off exponentially
  (up to 8s) on rate-limit or parse errors, and falls back across multiple
  keys via `txt2reward.llm.key_manager` if configured; if generation still fails, the
  previous reward program is kept.
- Training on CPU is slow; a GPU is strongly recommended.
- The sandbox forbids loops and attribute access, so generated reward
  functions must express any aggregation as a single arithmetic expression
  — a deliberate constraint, and sufficient for per-step reward shaping.
- Reward program updates happen in the main process only; worker processes
  see the update with a delay of up to `--reload-interval` steps.

---

## License

MIT — see [LICENSE](LICENSE).

---

## References

- [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Groq](https://console.groq.com)
- [Text2Reward paper](https://arxiv.org/abs/2309.11489)
