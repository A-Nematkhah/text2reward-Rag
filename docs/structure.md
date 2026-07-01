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
