"""Archive retrieval helpers, deduplication, and LLM formatting.

Public functions rank archive entries for RAG context, detect pathological
behaviours (crash farming, stationary farming), and format entries for prompts.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

from txt2reward.archive.curriculum import infer_curriculum_phase
from txt2reward.archive.fitness import (
    _V7_STATIONARY_SPEED_MAX,
    _comfort_score,
    _overtake_score,
    _passive_driving_gate,
    _safety_gate,
    _speed_score,
    _ttc_score,
    compute_fitness,
    is_passive_driving,
)
from txt2reward.config.fitness import CRASH_FARMING_CRASH_MIN, CRASH_FARMING_SPEED_MIN
from txt2reward.core.metrics import enrich_fitness_metrics


def reward_code_hash(code: str) -> str:
    """Stable 16-hex digest of reward source for deduplication."""
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()[:16]


def _structural_skeleton(code: str) -> str:
    """Replace numeric literals with a placeholder so near-duplicate reward
    functions (differing only in constant coefficients) collapse to the same
    skeleton hash, while leaving non-numeric structure intact."""
    return re.sub(r"-?\d+\.?\d*", "#", code.strip())


def reward_code_skeleton_hash(code: str) -> str:
    """Stable 16-hex digest of a reward function's structural skeleton
    (numeric literals normalised out) — use alongside reward_code_hash for
    near-duplicate detection, not as a replacement for exact-dedup."""
    return hashlib.sha256(_structural_skeleton(code).encode("utf-8")).hexdigest()[:16]


def effective_fitness(entry: Mapping[str, Any]) -> float:
    """
    Fitness used for archive ranking.  Always recomputed from metrics with the
    current default fitness version so stale stored scores (v7 flatlines or
    misleadingly high crash-farming fitness) do not dominate retrieval.
    """
    metrics = enrich_fitness_metrics(dict(entry.get("metrics", {})))
    gen = int(entry.get("generation", 0))
    return float(compute_fitness(metrics, generation=gen, prev_metrics=None))


def prefetch_effective_fitness(entries: Iterable[Mapping[str, Any]]) -> dict[int, float]:
    """Compute effective fitness once per generation for a batch of entries."""
    cache: dict[int, float] = {}
    for entry in entries:
        cached_effective_fitness(entry, cache)
    return cache


def cached_effective_fitness(entry: Mapping[str, Any], cache: dict[int, float]) -> float:
    """Return cached fitness for ``entry``, computing and storing on first use."""
    gen = int(entry.get("generation", -1))
    if gen not in cache:
        cache[gen] = effective_fitness(entry)
    return cache[gen]


def is_crash_farming(metrics: Mapping[str, Any]) -> bool:
    """Fast driving with near-universal crashes — local optimum to avoid."""
    return (
        float(metrics.get("crash_rate", 0.0)) >= CRASH_FARMING_CRASH_MIN
        and float(metrics.get("mean_speed", 0.0)) >= CRASH_FARMING_SPEED_MIN
    )


def is_stationary_farming(metrics: Mapping[str, Any]) -> bool:
    """True when mean speed indicates standing-still reward farming."""
    return float(metrics.get("mean_speed", 0.0)) < _V7_STATIONARY_SPEED_MAX


def is_pathological_for_retrieval(entry: Mapping[str, Any]) -> bool:
    """Entries that should not appear in top-k when healthier alternatives exist."""
    m = entry.get("metrics", {})
    return is_crash_farming(m) or is_stationary_farming(m)


def all_entries_crash_rate_above(
    entries: Sequence[Mapping[str, Any]],
    threshold: float,
) -> bool:
    """True when every entry's crash_rate is strictly above ``threshold``."""
    if not entries:
        return False
    return all(float(e.get("metrics", {}).get("crash_rate", 1.0)) > threshold for e in entries)


def dedupe_entries_by_code(
    entries: Sequence[Mapping[str, Any]],
    *,
    key: str = "reward_code",
) -> list[Mapping[str, Any]]:
    """Keep first occurrence of each unique reward program (by SHA-256 prefix)."""
    seen: set[str] = set()
    out: list[Mapping[str, Any]] = []
    for entry in entries:
        h = reward_code_hash(entry.get(key, ""))
        if h in seen:
            continue
        seen.add(h)
        out.append(entry)
    return out


def _format_entry(
    entry: Mapping[str, Any],
    show_code: bool,
    *,
    fitness: float | None = None,
) -> str:
    """Formats a single archive entry for LLM context."""
    m = entry["metrics"]
    cr = m.get("crash_rate", 0.5)
    meta = entry.get("critique_meta", {})
    fit = fitness if fitness is not None else effective_fitness(entry)
    lines = [
        f"--- Generation {entry['generation']} "
        f"(fitness={fit:.4f}) ---\n"
        f"Metrics:\n"
        f"  mean_speed     : {m.get('mean_speed', 0):.2f} m/s\n"
        f"  crash_rate     : {m.get('crash_rate', 0):.1%}\n"
        f"  mean_overtakes : {m.get('mean_overtakes', 0):.2f}/ep\n"
        f"  mean_ttc       : {m.get('mean_ttc', 0):.2f} s  "
        f"p10={m.get('p10_ttc', -1):.1f}s  min={m.get('min_ttc', -1):.1f}s\n"
        f"  mean_long_jerk : {m.get('mean_long_jerk', 0):.3f} m/s³\n"
        f"  completion_rate: {m.get('completion_rate', 0):.1%}\n"
        f"  mean_steps     : {m.get('mean_steps', 0):.0f}\n"
        f"  curriculum     : {m.get('curriculum_phase', infer_curriculum_phase(m))}\n"
        f"Fitness breakdown:\n"
        f"  speed_score    : {_speed_score(m.get('mean_speed', 0)):.3f}\n"
        f"  overtake_score : {_overtake_score(m.get('mean_overtakes', 0)):.3f}\n"
        f"  comfort_score  : {_comfort_score(m.get('mean_long_jerk', 0)):.3f}\n"
        f"  ttc_score      : {_ttc_score(m.get('mean_ttc', 30), m.get('p10_ttc', -1), m.get('min_ttc', -1)):.3f}\n"
        f"  safety_gate    : {_safety_gate(cr):.3f}\n"
        f"  passive_gate   : {_passive_driving_gate(m.get('mean_speed', 0), m.get('mean_overtakes', 0), cr):.3f}"
        f"{'  [PASSIVE — do not copy]' if is_passive_driving(m) else ''}\n"
        f"Failure modes: {meta.get('failure_modes', [])}\n"
        f"Strengths    : {meta.get('strengths', [])}\n"
    ]
    if meta.get("summary"):
        lines.append(f"Summary      : {meta['summary']}\n")
    if entry.get("critique"):
        # Show only first 300 chars of free-text critique to keep context tight
        crit_snippet = entry["critique"][:300].replace("\n", " ")
        lines.append(f"Critique     : {crit_snippet}...\n")
    if show_code:
        lines.append(f"Reward Code:\n```python\n{entry['reward_code']}\n```\n")
    return "\n".join(lines)
