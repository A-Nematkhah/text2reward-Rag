# txt2reward-v2

**True Text-to-Reward System for Highway Reinforcement Learning**

Train a highway driving agent with PPO where a language model (Groq / llama-3.3-70b)
**generates complete reward functions** (not just scalar weights), trains PPO, evaluates
behaviour, critiques the result (including reward-hacking detection), and evolves an
improved reward program — looping like an evolutionary search over reward code.

---

## Idea

Earlier versions of this project had the LLM tune a fixed set of weights (`w_speed`,
`w_safety`, ...) inside a hand-written formula. This version replaces that entirely:
the LLM now writes the formula itself, as a sandboxed Python function operating on a
structured state object.

```
Natural Language Goal
        │
        ▼
   RewardDesigner (main process)
        │  Groq API → llama-3.3-70b   (+ RAG context from reward_archive.json)
        ▼
  Generated compute_reward(state) source
        │  AST validation (no imports/exec/eval/loops/attribute access)
        ▼
  reward_program.py  ◄───────────────────────────────┐
        │                                              │
        ▼                                              │
  LLMRewardWrapper (per worker)                        │
  reloads reward_program.py every N steps ─────────────┘
        │
        ▼
  shaped_reward = compute_reward(state)
        │
        ▼
     PPO training
        │
        ▼
  evaluate_agent() → metrics (speed, crash rate, overtakes, completion)
        │
        ▼
  fitness = compute_fitness(metrics)
        │
        ▼
  LLM critique (reward hacking detection, failure modes, improvements)
        │
        ▼
  reward_archive.json  (generation, code, metrics, fitness, critique)
        │
        ▼
  Next generation (loop back to RewardDesigner)
```

---

## Project Structure

```
txt2reward-v2/
├── train.py               # Entry point — PPO training with the evolutionary loop
├── reward_wrapper.py       # Gym wrapper: loads + executes reward_program.py, collects stats
├── reward_designer.py      # LLM pipeline: generate / critique / evolve reward programs
├── reward_sandbox.py        # Secure sandbox: AST validation + restricted execution
├── reward_archive.py        # Persistent archive + fitness function + RAG retrieval
├── reward_program.py        # Current generated reward function (hot-swapped at runtime)
├── reward_components.py     # LEGACY — weight-based reward kept for evaluate.py --no-shaped
├── evaluate.py             # Evaluate a trained model against any generation's reward
├── plot_training.py         # Generates training/evolution/fitness charts
├── training_logger.py       # Persists per-episode + per-generation training history
├── requirements.txt        # Python dependencies
└── colab_setup.ipynb        # Ready-to-run Google Colab notebook
```

---

## Requirements

- Python 3.10+
- [Groq API key](https://console.groq.com) — free tier is sufficient
- GPU recommended (Colab T4 works fine; CPU is slow)

```bash
pip install -r requirements.txt
```

---

## Quickstart on Google Colab

**Fastest path:** open `colab_setup.ipynb`.

1. Set the runtime to GPU (Runtime → Change runtime type → T4 GPU)
2. Paste your `GROQ_API_KEY` when prompted
3. Run cells in order

---

## Local Setup

```bash
export GROQ_API_KEY="gsk_xxxxxxxx"

# Train with a natural-language goal (bootstraps reward_program.py automatically)
python train.py --timesteps 200000 --n-envs 4 \
  --goal "Drive fast and safely, overtake slow vehicles, avoid collisions, minimise harsh braking."

# Evaluate the trained model with the current reward program
python evaluate.py --model ppo_highway_txt2reward.zip --episodes 10

# Evaluate against a specific earlier generation from the archive
python evaluate.py --model ppo_highway_txt2reward.zip --generation 2
```

---

## CLI Reference

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--timesteps` | `200000` | Total environment steps |
| `--n-envs` | `4` | Number of parallel environments |
| `--reload-interval` | `200` | Steps between `reward_program.py` reloads in each worker |
| `--evolve-every` | `20` | Generate a new reward program every N episodes (after warmup) |
| `--warmup-episodes` | `40` | Episodes before the first LLM reward generation |
| `--goal` | (driving goal) | Natural language goal sent to the LLM |
| `--reward-path` | `reward_program.py` | Output path for the generated reward program |
| `--archive-file` | `reward_archive.json` | Path to the reward archive |
| `--bootstrap` | off | Force-generate an initial reward program before training |
| `--resume` | `None` | Path to a `.zip` checkpoint to resume from |
| `--drive-dir` | `/content/drive/MyDrive/txt2reward` | Google Drive folder for sync |
| `--checkpoint-freq` | `10000` | Steps between checkpoints |

### `evaluate.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `ppo_highway_txt2reward.zip` | Path to trained model |
| `--episodes` | `10` | Number of evaluation episodes |
| `--no-shaped` | off | Disable shaped reward (env reward only) |
| `--reward-path` | `reward_program.py` | Reward program to evaluate with |
| `--generation` | `None` | Evaluate using a specific archived generation |
| `--render` | off | Render the environment visually |
| `--stochastic` | off | Use stochastic policy (default: deterministic) |
| `--save` | `None` | Save results as JSON to this path |

---

## How It Works

### State Object

Every step, the environment observation is parsed into a structured state dict
passed to the generated reward function:

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

### Reward Program

The LLM generates a complete `compute_reward(state) -> float` function (see
`reward_program.py` for the current/default example). It is sandboxed:

- No `import`, no `eval`/`exec`, no file/network access, no attribute access, no loops.
- Only approved math (`sqrt`, `exp`, `log`, trig, `clip`, etc.) and approved state keys.
- Validated via AST inspection (`reward_sandbox.validate_reward_code`) before being
  written to disk or executed.

### Evolutionary Loop

1. **Warmup phase** — first N episodes train with the current/default reward program.
2. **Every `evolve_every` episodes** after warmup:
   - Episode statistics are aggregated into evaluation metrics.
   - The **current** reward program + metrics are sent to the LLM for **critique**
     (reward hacking detection, failure modes, proposed fixes).
   - The critique and metrics are stored in `reward_archive.json` alongside the
     program's `fitness` score.
   - The LLM is asked to **generate an improved reward program**, given top-performing
     archive entries as RAG-style context.
   - The new program is validated; on success it replaces `reward_program.py` and the
     generation counter increments. On failure, the previous program is kept.
3. Worker environments reload `reward_program.py` every `reload-interval` steps.

### Fitness Function

```
fitness = ( w_speed·speed_score + w_overtake·overtake_score
          + w_lane·lane_score   + w_safety·safety_score
          + w_complete·completion_score ) × exp(-5 · crash_rate)
```

The exponential term sharply penalises high crash rates regardless of other metrics.

### Reward Hacking Detection

The critique prompt explicitly asks the LLM to look for:

- Oscillatory lane changes (many lane changes, few overtakes)
- Acceleration spam / brake-acceleration exploits (high jerk, no speed gain)
- Stationary reward farming (low speed, high shaped reward)
- TTC exploitation (very low TTC without crashes — tailgating for some bonus)

### Multi-Process Architecture

Training uses `SubprocVecEnv` — each worker runs in a separate process. The shared
state is the `reward_program.py` file on disk:

- **Main process** — runs `RewardDesigner`, calls Groq, validates and writes the new
  reward program to disk, maintains `reward_archive.json`.
- **Worker processes** — run `LLMRewardWrapper`, periodically reload and execute the
  reward program from disk.

---

## Output Files

| File | Description |
|---|---|
| `ppo_highway_txt2reward.zip` | Final trained PPO model |
| `ppo_highway_*.zip` | Intermediate checkpoints |
| `reward_program.py` | Current generated reward function |
| `reward_archive.json` | Every generation: code, metrics, fitness, critique |
| `training_log.json` | Per-episode and per-generation training history |
| `tb_logs/` | TensorBoard training logs |

---

## Bug Fixes (Generation-Tracking & Reward-Hacking Detection)

A real training run surfaced four related issues, all fixed in this version:

1. **Archive stopped growing after generation 0.** `_evolve()` used an
   independent `self._generation` counter alongside `len(archive.entries)`.
   If `add_entry()` was ever skipped (e.g. because the on-disk reward file
   was momentarily empty), the counter kept incrementing forever while the
   archive silently froze at 1 entry — so critiques were always written
   against a stale generation 0, and RAG context never improved.
   **Fix:** `generation` is now a property derived solely from
   `len(self.archive.entries)`. `_evolve()` archives the program that just
   ran unconditionally (with a loud warning + safe placeholder if the code
   is somehow missing, instead of silently skipping), then critiques the
   entry it just created — never a stale reference.

2. **Bootstrap reward was never archived.** `train.py`'s bootstrap step
   generated `reward_program.py` via a throwaway `bootstrap_designer` that
   never called `archive.add_entry()`. **Fix:** there is now a single
   `RewardDesigner` instance for the whole run; the first real `_evolve()`
   call archives the bootstrap code together with its first real measured
   metrics.

3. **Drive restore ran after bootstrap.** This meant a freshly-bootstrapped
   `reward_program.py` could be immediately overwritten by a stale Drive
   copy. **Fix:** restore-from-Drive now runs first; bootstrap only fires if
   no usable reward program exists locally or on Drive. A new reconciliation
   step in `RewardDesigner.__init__` also self-heals partial restores (e.g.
   archive restored but `reward_program.py` missing) by rewriting the reward
   file from the archive's latest entry.

4. **Undetected reward hacking ("slow down to avoid crashing").** Because
   critiques were stuck on generation 0, the LLM never saw that mean speed
   and overtakes were declining release-over-release while crash rate
   improved — a classic reward-hacking pattern where the agent learns
   passive/stationary behaviour instead of driving well. **Fix:** the
   critique prompt now includes an explicit `TREND VS PREVIOUS GENERATION`
   section with deltas, and a hard-coded warning fires whenever speed or
   overtakes drop while crash rate improves, telling the LLM to call this
   out explicitly.

---



If you used the previous (weight-tuning) version of this project:

1. `reward_weights.json` is no longer the primary mechanism — it is replaced by
   `reward_program.py`. `reward_components.py` is kept only as a legacy compatibility
   layer for `evaluate.py --no-shaped`.
2. `RewardDesigner` no longer exposes `get_weights()` returning a weight dict; it now
   returns `{"generation": int, "reward_path": str}` for logging compatibility.
3. Existing training logs (`training_log.json`) from the old format are not compatible
   with the new per-episode schema (`generation` replaces weight snapshots) — start a
   fresh log or migrate manually.
4. Re-run with `--bootstrap` on first launch to generate an initial LLM-written reward
   program instead of relying on the bundled default in `reward_program.py`.

---

## Known Limitations

- Groq free tier is rate-limited. The designer uses exponential backoff (up to 8s) on
  rate-limit/parse errors and keeps the previous reward program if generation fails.
- Training on CPU is slow; a GPU is strongly recommended.
- The sandbox forbids loops and attribute access, so generated reward functions must
  express any iteration as a single arithmetic expression (sufficient for per-step
  reward shaping, but a deliberate constraint).
- Reward program updates happen in the main process only — workers see the updated
  program with a delay of up to `reload-interval` steps.

---

## References

- [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Groq](https://console.groq.com)
- [Text2Reward paper](https://arxiv.org/abs/2309.11489)
