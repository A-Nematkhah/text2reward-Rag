"""Structured critique metadata parsing for archive entries."""

from __future__ import annotations

import json
from typing import Any, Mapping, cast

from txt2reward.archive.fitness import is_passive_driving
from txt2reward.core.types import CritiqueMeta, FitnessMetrics

# ── Structured failure mode detection (improvement #4) ───────────────────────

# Known failure-mode tag names (used for retrieval in improvement #3)
FAILURE_MODE_TAGS = frozenset(
    {
        "tailgating",
        "passive_driving",
        "oscillatory_lane_changes",
        "acceleration_spam",
        "stationary_farming",
        "reward_hacking",
    }
)

STRENGTH_MODE_TAGS = frozenset(
    {
        "high_speed",
        "good_overtaking",
        "safe_driving",
        "smooth_driving",
    }
)


def _filter_known_tags(tags: Any, allowed: frozenset[str]) -> list[str]:
    if not isinstance(tags, list):
        return []
    return [t for t in tags if isinstance(t, str) and t in allowed]


def _extract_critique_meta_json(critique_text: str) -> dict[str, Any] | None:
    """Parse CRITIQUE_META:{...} using balanced-brace extraction."""
    marker = "CRITIQUE_META:"
    if marker not in critique_text:
        return None
    raw = critique_text[critique_text.index(marker) + len(marker) :].strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_structured_critique(
    critique_text: str,
    metrics: FitnessMetrics | Mapping[str, Any],
) -> CritiqueMeta:
    """
    Improvement #4: produce machine-readable critique metadata from the free-text
    critique + metrics.  This is called inside update_critique() and also used
    by the LLM-critique JSON path (reward_designer.py).

    Heuristic rules fire on metrics so we always have SOME structured data
    even when the LLM critique is unavailable.  The LLM JSON path overwrites
    these if a valid JSON block is present in critique_text.

    Returns a dict:
      {
        "failure_modes": list[str],
        "strengths":     list[str],
        "summary":       str,
      }
    """
    failure_modes: list[str] = []
    strengths: list[str] = []

    mean_speed = float(metrics.get("mean_speed", 0.0))
    crash_rate = float(metrics.get("crash_rate", 0.5))
    mean_ttc = float(metrics.get("mean_ttc", 30.0))
    min_ttc = float(metrics.get("min_ttc", -1.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    mean_jerk = float(metrics.get("mean_long_jerk", 0.0))
    mean_accel = float(metrics.get("mean_accel", 0.0))
    lc = int(metrics.get("total_lane_changes", 0))
    ot = float(metrics.get("mean_overtakes", 0.0))

    # Heuristic failure mode tagging
    if mean_speed < 5.0:
        failure_modes.append("stationary_farming")
    if is_passive_driving(metrics):
        if "stationary_farming" not in failure_modes:
            failure_modes.append("passive_driving")
    effective_ttc = min_ttc if min_ttc >= 0 else mean_ttc
    if effective_ttc < 2.0 and crash_rate < 0.2:
        failure_modes.append("tailgating")
    n_eps = max(int(metrics.get("n_episodes", 1)), 1)
    if lc > 0 and ot > 0 and lc / max(ot * n_eps, 1) > 5:
        failure_modes.append("oscillatory_lane_changes")
    if mean_jerk > 2.5 or mean_accel > 3.0:
        failure_modes.append("acceleration_spam")

    # Heuristic strength tagging
    if mean_speed >= 26.0:
        strengths.append("high_speed")
    if mean_overtakes >= 3.0:
        strengths.append("good_overtaking")
    if crash_rate <= 0.05:
        strengths.append("safe_driving")
    if mean_jerk <= 0.5:
        strengths.append("smooth_driving")

    # Try to extract JSON block if the LLM embedded one in the critique text
    # (reward_designer.py can inject a JSON block at the end of the critique)
    meta: dict[str, Any] = {
        "failure_modes": failure_modes,
        "strengths": strengths,
        "summary": "",
    }
    if critique_text:
        parsed = _extract_critique_meta_json(critique_text)
        if parsed:
            llm_failures = _filter_known_tags(parsed.get("failure_modes"), FAILURE_MODE_TAGS)
            if llm_failures:
                meta["failure_modes"] = llm_failures
            llm_strengths = _filter_known_tags(parsed.get("strengths"), STRENGTH_MODE_TAGS)
            if llm_strengths:
                meta["strengths"] = llm_strengths
            if isinstance(parsed.get("summary"), str):
                meta["summary"] = parsed["summary"]

    if not meta["summary"]:
        if failure_modes:
            meta["summary"] = f"Detected issues: {', '.join(failure_modes)}."
        elif strengths:
            meta["summary"] = f"Good performance: {', '.join(strengths)}."
        else:
            meta["summary"] = "No major issues detected."

    return cast(CritiqueMeta, meta)
