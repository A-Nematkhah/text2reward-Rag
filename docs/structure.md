# Repository layout

```
text2reward-Rag/
├── train.py / evaluate.py / plot_training.py   # CLI entry points
├── reward_program.py                         # Active reward (hot-reloaded)
├── txt2reward/
│   ├── core/
│   │   └── metrics.py          # percentile, TTC pooling, enrich_fitness_metrics
│   ├── config/
│   │   ├── env.py              # highway-v0 ENV_CONFIG
│   │   ├── paths.py            # artifact paths (reward, archive, log)
│   │   ├── training.py         # PPO + evolution schedule defaults
│   │   ├── llm.py              # model + generation hyperparameters
│   │   ├── fitness.py          # fitness version + archive retrieval thresholds
│   │   └── validation.py       # smoke-test + trajectory-bank gates
│   ├── archive/
│   │   ├── archive.py          # Backward-compatible re-export facade
│   │   ├── store.py            # RewardArchive CRUD + persistence
│   │   ├── fitness.py          # compute_fitness v6/v7/v8
│   │   ├── curriculum.py       # metrics-driven curriculum phases
│   │   ├── critique.py         # structured critique metadata
│   │   └── retrieval.py        # RAG formatting, dedup, effective_fitness
│   ├── llm/
│   │   ├── designer.py         # RewardDesigner evolution orchestration
│   │   ├── prompts.py          # LLM prompt templates
│   │   ├── validation.py       # smoke-test + Stage B pipeline
│   │   ├── aggregation.py      # episode → archive metrics
│   │   └── key_manager.py      # Groq API key rotation
│   ├── sandbox/
│   │   └── sandbox.py          # AST validation + safe execution
│   ├── trajectory/
│   │   └── bank.py             # synthetic trajectory bank (Stage B)
│   ├── reward/
│   │   ├── wrapper.py          # LLMRewardWrapper
│   │   └── components.py       # legacy weight-based reward
│   ├── training/
│   │   ├── train.py            # main() orchestration only
│   │   ├── env_factory.py      # make_env, build_vec_env
│   │   ├── callbacks.py        # RewardEvolutionCallback
│   │   ├── device.py           # detect_device()
│   │   ├── logger.py           # TrainingLogger
│   │   └── plots.py            # training charts
│   └── evaluation/
│       └── evaluate.py         # post-training evaluation
├── scripts/
│   └── calibrate_smoke_gate.py # Stage B gate calibration (offline)
├── examples/
│   └── README.md               # CLI workflow templates
├── tests/
│   ├── safety/                 # Malicious reward + security regression tests
│   └── test_*.py               # Unit / integration tests (115 total)
└── docs/
    ├── structure.md            # Module layout and responsibilities
    └── scripts.md              # Script index (mirrors scripts/README.md)
```

## Module responsibilities

| Module | Owns | Does not own |
|--------|------|----------------|
| `archive/store.py` | persistence, CRUD, `format_for_llm` | fitness math, smoke tests |
| `archive/fitness.py` | fitness scoring | LLM, training |
| `llm/designer.py` | evolve / critique / generate orchestration | validation internals, prompts |
| `llm/validation.py` | smoke-test gates | archive, PPO |
| `training/train.py` | CLI wiring | env logic, callbacks, device |
| `core/metrics.py` | shared metric helpers | fitness weights |
| `config/` | centralized hyperparameters and paths | business logic |
| `sandbox/` | AST validation + timed execution of LLM rewards | training, archive |

Public APIs use module/class docstrings documenting purpose, inputs, outputs,
and side effects where non-obvious (disk writes, network calls).

Security: generated reward code is AST-validated (no imports/loops/attributes),
executed in a restricted namespace with wall-clock timeouts, re-validated on
archive restore, and never written to disk without passing smoke tests.

Dependencies: runtime pins in `requirements.txt`; `pytest` in `requirements-dev.txt`.

Repository hygiene: `LICENSE` (MIT), `api_keys.json.example`, `.gitignore` for
runtime artifacts and caches, `examples/README.md` for CLI workflows.
