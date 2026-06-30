# Debug backlog — training / evolution issues

Recorded after analysis of run with 26 generations, 100% crash rate, fitness ~0.002.

---

## Task 1 — Lite trajectory bank for Stage B gate — **DONE**

Implemented: `build_trajectory_bank_lite()` (16 trajectories), `get_trajectory_bank()`, `TRAJECTORY_BANK_MODE=lite|full` (default **lite**), wired into `txt2reward/llm/validation.py`. Full bank remains for `scripts/calibrate_smoke_gate.py` and explicit tests.

**Problem:** Full bank has 40 trajectories → ~458 soft pairwise checks at 13% threshold (~59 violations max). LLM repair attempts often fail at 13.3–19% (2–30 pairs over margin) or with passive violations (`passive=12`). Bootstrap passes at ~6.6%; accepted gen-26 passes at ~10% — gate is tight, not broken.

**Goal:** Reduce false rejections during evolution without removing safety intent.

**Proposed work:**

1. Add `build_trajectory_bank_lite()` in `txt2reward/trajectory/bank.py` — ~16 trajectories (1–2 variants per category × 8 categories).
2. Config flag in `txt2reward/config/validation.py`, e.g. `TRAJECTORY_BANK_MODE = "lite" | "full"` (default `"lite"` for evolution, `"full"` for calibration/audit).
3. Wire `_full_validation_pipeline` in `txt2reward/llm/validation.py` to use lite bank when configured.
4. Tests:
   - Bootstrap and shipped default still PASS lite bank.
   - Lite bank has fewer decisive pairs; same categories still covered.
   - Document expected soft-pair count (~100–120 vs ~458).
5. Optional: run `scripts/calibrate_smoke_gate.py` (or new script) comparing acceptance rate on last N archived rejected codes.

**Success criteria:**

- LLM acceptance rate on repair loop increases (fewer “evolution skipped — smoke fail”).
- Bootstrap + default reward still PASS.
- No regression on `tests/test_validation.py` / trajectory bank tests.

**Out of scope for this task:** Changing PPO training or reward scale (see Task 2).

---

## Task 2 — Reward scale / evolve gating (priority: critical for crash fix)

**Problem:** 100% crash from gen 0; LLM increases penalty magnitudes each generation → `ep_rew_mean` -1.2k → -3.2k, `value_loss` ~100k → ~400k, `explained_variance ≈ 0`. Smoke-passing rewards still produce 100% crash in env.

**Done (partial):**

- [x] Reward step clipping in `LLMRewardWrapper` — clip per-step to `[-10, 10]` (`REWARD_STEP_CLIP_MIN/MAX` in `config/training.py`).
- [x] Increase default `--evolve-every` to **100** and `--warmup-episodes` to **80**.

**Proposed work (remaining):**

- [ ] Skip evolution when `crash_rate >= 0.90` (train more on fixed reward first).
- [ ] Reset to bootstrap reward and freeze evolution until crash_rate improves.
- [ ] Optional: easier env for phase 1 (`vehicles_count=15`).

---

## Task 3 — Archive RAG when all entries are crash-farmers (priority: medium)

**Problem:** `best fitness=0.0021` with 100% crash — top-k retrieval surfaces fast crash farmers as “top performers”.

**Proposed work:**

- [ ] Verify curriculum-aware retrieval + pathological filter behave as intended under 100% crash archive.
- [ ] Consider suppressing top-k examples when all entries have `crash_rate > 0.5`.

---

## Reference numbers

| Metric | Full bank | Lite bank (default) |
|--------|-----------|---------------------|
| Trajectories | 40 | 16 |
| Soft decisive pairs (bootstrap) | ~458 | ~80–100 (run tests) |
| `BANK_MAX_VIOLATION_RATE` | 0.13 | 0.13 |
| Bootstrap soft rate (full, measured) | ~6.6% | (see tests) |

Legacy full-bank reference:

| Metric | Value |
|--------|-------|
| Decisive pairs | ~612 |
| Max soft violations allowed @ 13% | ~59 |
| Typical failed LLM attempt (full) | 13.3–19% or passive > 0 |
