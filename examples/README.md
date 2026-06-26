# Examples

This repository uses thin CLI entry points at the repo root rather than a separate
examples package. Typical workflows:

## Training with evolution

```bash
export GROQ_API_KEY="gsk_xxxxxxxx"
python train.py --timesteps 200000 --n-envs 4 \
  --goal "Drive fast and safely, overtake slow vehicles, avoid collisions."
```

## Evaluate a checkpoint

```bash
python evaluate.py --model ppo_highway_txt2reward.zip --episodes 10
```

## Plot training history

```bash
python plot_training.py
```

## API key pool (optional)

Copy `api_keys.json.example` to `api_keys.json` at the repo root and list multiple
Groq keys for automatic rotation on rate limits.

## Reward program template

See `reward_program.py` for the active `compute_reward(state)` function shape that
the LLM must generate. Generated programs are validated by `txt2reward.sandbox` and
`txt2reward.llm.validation` before being written to disk.
