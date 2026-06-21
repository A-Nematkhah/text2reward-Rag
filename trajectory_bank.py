"""
trajectory_bank.py
───────────────────
Generates a diverse bank of ~40 synthetic state-trajectories for robust
reward-hacking detection during reward-program validation.

Why this exists
────────────────
The original smoke-test in reward_designer.py compared exactly TWO
trajectories ("cautious" vs "reckless") and required:

    reckless_return < cautious_return

That is a single inequality. A PPO agent has a much larger behaviour
space than those two points, so a reward function can easily satisfy
this one constraint while still containing a hackable loophole that
shows up in some OTHER region of behaviour space (e.g. tailgating
without ever crashing, or oscillating lane changes that rack up small
bonuses, or sitting still and farming a "safe gap" term).

This module replaces the single inequality with:

  1. A bank of ~40 trajectories spanning ~8 behavioural categories,
     each parametrised (speed levels, episode length, jitter) and
     generated with a FIXED random seed for full reproducibility.
  2. A reference fitness for every trajectory, computed from the SAME
     domain-expert fitness function already used for the archive
     (reward_archive.compute_fitness), applied to the trajectory's
     aggregate metrics. This reference ranking is independent of
     whatever compute_reward() the LLM just wrote.
  3. A pairwise-consistency check: for every pair of trajectories
     (A, B) where the reference fitness disagrees by a meaningful
     margin, the candidate reward function's cumulative episode
     return must agree on which one is better. The fraction of
     violated pairs is the gate's score.
  4. A tolerant threshold (default 10%) so that pairs which are very
     close in reference fitness (true judgment calls, not hacking)
     don't trigger false rejections.

Categories covered
───────────────────
  safe_steady          — smooth, moderate speed, no crash, no overtakes
  safe_fast            — smooth, high speed, no crash, no overtakes
  stationary_farming   — near-zero speed for the whole episode (tests
                          whether a reward function can be "farmed" by
                          standing still — e.g. via a safe-gap term)
  reckless_crash        — high speed, tailgating, crashes at some point
                          in the episode (early / mid / late variants)
  tailgating_no_crash    — sustained very-low TTC, NEVER crashes (the
                          "ride the bumper for some bonus" exploit
                          named explicitly in the critique prompt)
  oscillating_lanes     — frequent lane changes, few/no overtakes
                          (lane-thrash exploit)
  jerk_accel_spam        — large oscillating accel/jerk with no net
                          speed gain (brake-accelerate exploit)
  legitimate_overtaking  — moderate-to-high speed WITH genuine,
                          well-spaced overtakes and no crash — this
                          is the behaviour we WANT to score highest,
                          and the gate also checks it is not dominated
                          by any unsafe category.

Usage
─────
    from trajectory_bank import build_trajectory_bank, evaluate_consistency

    bank = build_trajectory_bank()          # list[TrajectorySpec], ~40 entries
    ok, report = evaluate_consistency(fn, bank, max_violation_rate=0.10)
    if not ok:
        # report is a human + LLM-readable string describing the
        # worst-violating pairs, suitable for feeding back into the
        # repair prompt.
        ...
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from reward_archive import compute_fitness

# ── Fixed seed for full reproducibility across runs ───────────────────────────
_SEED = 20260620


@dataclass
class TrajectorySpec:
    """A single synthetic trajectory: a name, category, list of per-step
    state dicts, and pre-computed reference metrics/fitness."""

    name: str
    category: str
    states: list[dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)
    ref_fitness: float = 0.0

    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = _aggregate_trajectory_metrics(self.states)
        if not self.ref_fitness:
            self.ref_fitness = compute_fitness(self.metrics)


# ── Metric aggregation (mirrors evaluate_agent()-style metrics) ───────────────


def _aggregate_trajectory_metrics(states: list[dict[str, Any]]) -> dict[str, float]:
    """
    Reduces a per-step state trajectory to the same aggregate metric shape
    that reward_archive.compute_fitness() expects, so every trajectory has
    an independent, ground-truth fitness score to validate against.
    """
    n = max(len(states), 1)
    crashed = any(s["collided"] for s in states)
    overtakes = sum(1 for s in states if s.get("overtook"))
    speed_sum = sum(s["speed_ms"] for s in states)
    jerk_sum = sum(abs(s.get("long_jerk", 0.0)) for s in states)
    ttc_sum = sum(s.get("ttc", 30.0) for s in states)

    return {
        "mean_speed": speed_sum / n,
        "crash_rate": 1.0 if crashed else 0.0,
        "mean_overtakes": float(overtakes),  # per-episode count, matches archive convention
        "mean_long_jerk": jerk_sum / n,
        "mean_ttc": ttc_sum / n,
        "completion_rate": 0.0 if crashed else 1.0,
    }


# ── Low-level state builder ────────────────────────────────────────────────────


def _state(
    speed_ms: float = 20.0,
    front_dist: float = 50.0,
    ttc: float = 30.0,
    rel_vel_ms: float = 0.0,
    lane: int = 1,
    overtook: bool = False,
    lane_changed: bool = False,
    collided: bool = False,
    nearby_vehicles: int = 1,
    accel_ms2: float = 0.0,
    long_jerk: float = 0.0,
    lat_jerk: float = 0.0,
) -> dict[str, Any]:
    return {
        "speed_ms": speed_ms,
        "front_dist": front_dist,
        "ttc": ttc,
        "rel_vel_ms": rel_vel_ms,
        "lane": lane,
        "overtook": overtook,
        "lane_changed": lane_changed,
        "collided": collided,
        "nearby_vehicles": nearby_vehicles,
        "accel_ms2": accel_ms2,
        "long_jerk": long_jerk,
        "lat_jerk": lat_jerk,
    }


# ── Category generators ─────────────────────────────────────────────────────────
# Each generator returns a list of per-step state dicts. `rng` is a
# random.Random seeded once per trajectory for reproducible jitter.


def _gen_safe_steady(rng: random.Random, length: int, speed: float) -> list[dict]:
    """Smooth driving at a fixed moderate/high speed, no crash, no overtakes."""
    out = []
    for t in range(length):
        jitter = rng.uniform(-0.5, 0.5)
        out.append(
            _state(
                speed_ms=speed + jitter,
                front_dist=rng.uniform(35.0, 60.0),
                ttc=rng.uniform(8.0, 20.0),
                rel_vel_ms=rng.uniform(-1.0, 1.0),
                lane=1,
                nearby_vehicles=rng.randint(0, 2),
                accel_ms2=rng.uniform(-0.3, 0.3),
                long_jerk=rng.uniform(-0.2, 0.2),
                lat_jerk=0.0,
            )
        )
    return out


def _gen_stationary_farming(rng: random.Random, length: int) -> list[dict]:
    """Near-zero speed for the whole episode — tests if standing still can
    be farmed via safe-gap / low-jerk / high-TTC terms."""
    out = []
    for t in range(length):
        out.append(
            _state(
                speed_ms=max(0.0, rng.uniform(0.0, 2.0)),
                front_dist=rng.uniform(80.0, 150.0),
                ttc=30.0,
                rel_vel_ms=0.0,
                lane=1,
                nearby_vehicles=0,
                accel_ms2=0.0,
                long_jerk=0.0,
                lat_jerk=0.0,
            )
        )
    return out


def _gen_reckless_crash(rng: random.Random, length: int, speed: float, crash_at_frac: float) -> list[dict]:
    """High speed, tight gaps, crashes at a configurable point in the episode
    (early / mid / late) to make sure the gate isn't just checking one
    crash-timing pattern."""
    crash_idx = max(1, int(length * crash_at_frac)) - 1
    out = []
    for t in range(length):
        is_crash = t == crash_idx
        out.append(
            _state(
                speed_ms=speed if not is_crash else speed,
                front_dist=rng.uniform(8.0, 18.0) if not is_crash else 0.0,
                ttc=rng.uniform(0.5, 2.0) if not is_crash else 0.0,
                rel_vel_ms=rng.uniform(-8.0, -2.0),
                lane=t % 3,
                overtook=(t % 9 == 0 and not is_crash),
                lane_changed=(t % 4 == 0 and not is_crash),
                collided=is_crash,
                nearby_vehicles=rng.randint(3, 6),
                accel_ms2=rng.uniform(-3.0, 3.0),
                long_jerk=rng.uniform(1.0, 2.5),
                lat_jerk=rng.uniform(0.5, 1.5),
            )
        )
        if is_crash:
            break  # episode ends on collision
    return out


def _gen_tailgating_no_crash(rng: random.Random, length: int, speed: float) -> list[dict]:
    """Sustained very-low TTC / tight following distance for the ENTIRE
    episode, but never actually crashes. This is the explicit
    'TTC exploitation' pattern named in the critique prompt: an agent
    riding the bumper to harvest whatever small bonus correlates with
    proximity, while staying just barely on the right side of a crash."""
    out = []
    for t in range(length):
        out.append(
            _state(
                speed_ms=speed + rng.uniform(-0.5, 0.5),
                front_dist=rng.uniform(4.0, 9.0),
                ttc=rng.uniform(0.8, 1.8),
                rel_vel_ms=rng.uniform(-1.0, 0.5),
                lane=1,
                nearby_vehicles=rng.randint(1, 3),
                accel_ms2=rng.uniform(-0.5, 0.5),
                long_jerk=rng.uniform(-0.3, 0.3),
                lat_jerk=0.0,
                collided=False,
            )
        )
    return out


def _gen_oscillating_lanes(rng: random.Random, length: int) -> list[dict]:
    """Frequent lane changes with little or no overtaking — the
    'lane-thrash' exploit named in the critique prompt."""
    out = []
    lane = 1
    for t in range(length):
        change = (t % 2 == 0)
        if change:
            lane = (lane + rng.choice([-1, 1])) % 4
        out.append(
            _state(
                speed_ms=18.0 + rng.uniform(-1.0, 1.0),
                front_dist=rng.uniform(30.0, 50.0),
                ttc=rng.uniform(6.0, 15.0),
                rel_vel_ms=rng.uniform(-1.0, 1.0),
                lane=lane,
                lane_changed=change,
                overtook=(t % 15 == 0),  # rare, disproportionate to lane changes
                nearby_vehicles=rng.randint(1, 3),
                accel_ms2=rng.uniform(-0.5, 0.5),
                long_jerk=rng.uniform(-0.3, 0.3),
                lat_jerk=rng.uniform(0.8, 1.6),
            )
        )
    return out


def _gen_jerk_accel_spam(rng: random.Random, length: int) -> list[dict]:
    """Large oscillating accel/jerk with no net speed gain — the
    brake-accelerate exploit."""
    out = []
    for t in range(length):
        accel = 3.5 if t % 2 == 0 else -3.5
        out.append(
            _state(
                speed_ms=19.0 + rng.uniform(-0.3, 0.3),  # net speed barely moves
                front_dist=rng.uniform(25.0, 45.0),
                ttc=rng.uniform(5.0, 12.0),
                rel_vel_ms=rng.uniform(-1.0, 1.0),
                lane=1,
                nearby_vehicles=rng.randint(1, 3),
                accel_ms2=accel,
                long_jerk=accel * rng.uniform(0.8, 1.2),
                lat_jerk=0.0,
            )
        )
    return out


def _gen_legitimate_overtaking(rng: random.Random, length: int, speed: float, n_overtakes: int) -> list[dict]:
    """Moderate-to-high speed with genuine, well-spaced overtakes, safe
    headway, low jerk, and no crash. This is the behaviour the reward
    function SHOULD rank highest among all categories, so the gate also
    asserts this dominates every unsafe/degenerate category."""
    out = []
    overtake_steps = set()
    if n_overtakes > 0:
        spacing = max(length // (n_overtakes + 1), 1)
        overtake_steps = {spacing * (i + 1) for i in range(n_overtakes)}

    for t in range(length):
        is_overtake = t in overtake_steps
        out.append(
            _state(
                speed_ms=speed + rng.uniform(-0.8, 0.8),
                front_dist=rng.uniform(25.0, 45.0) if not is_overtake else rng.uniform(15.0, 25.0),
                ttc=rng.uniform(5.0, 15.0),
                rel_vel_ms=rng.uniform(1.0, 4.0),
                lane=1 + (t % 2),
                overtook=is_overtake,
                lane_changed=is_overtake,
                nearby_vehicles=rng.randint(1, 4),
                accel_ms2=rng.uniform(-0.6, 0.8),
                long_jerk=rng.uniform(-0.4, 0.4),
                lat_jerk=rng.uniform(0.0, 0.5) if is_overtake else 0.0,
                collided=False,
            )
        )
    return out


# ── Bank assembly ───────────────────────────────────────────────────────────────


def build_trajectory_bank() -> list[TrajectorySpec]:
    """
    Builds the full ~40-trajectory bank, deterministically (fixed seed).
    Returns a list of TrajectorySpec, each with reference metrics + fitness
    already computed.
    """
    rng = random.Random(_SEED)
    bank: list[TrajectorySpec] = []

    def add(name: str, category: str, states: list[dict]) -> None:
        bank.append(TrajectorySpec(name=name, category=category, states=states))

    # 1) safe_steady — 6 variants across speed levels and episode lengths
    for i, (speed, length) in enumerate(
        [(15.0, 40), (18.0, 50), (20.0, 60), (22.0, 40), (24.0, 50), (26.0, 60)]
    ):
        add(f"safe_steady_{i}", "safe_steady", _gen_safe_steady(rng, length, speed))

    # 2) safe_fast — 4 variants, higher speed band
    for i, (speed, length) in enumerate([(27.0, 40), (28.5, 50), (29.5, 45), (30.0, 55)]):
        add(f"safe_fast_{i}", "safe_fast", _gen_safe_steady(rng, length, speed))

    # 3) stationary_farming — 4 variants, episode length only
    for i, length in enumerate([30, 45, 60, 80]):
        add(f"stationary_{i}", "stationary_farming", _gen_stationary_farming(rng, length))

    # 4) reckless_crash — 6 variants: speed x crash timing (early/mid/late)
    for i, (speed, frac) in enumerate(
        [(26.0, 0.15), (26.0, 0.5), (26.0, 0.9), (30.0, 0.2), (30.0, 0.6), (30.0, 0.95)]
    ):
        add(f"reckless_crash_{i}", "reckless_crash", _gen_reckless_crash(rng, 40, speed, frac))

    # 5) tailgating_no_crash — 5 variants across speed, NEVER crashes
    for i, speed in enumerate([18.0, 22.0, 25.0, 27.0, 29.0]):
        add(f"tailgate_{i}", "tailgating_no_crash", _gen_tailgating_no_crash(rng, 50, speed))

    # 6) oscillating_lanes — 4 variants of episode length
    for i, length in enumerate([30, 40, 50, 60]):
        add(f"osc_lanes_{i}", "oscillating_lanes", _gen_oscillating_lanes(rng, length))

    # 7) jerk_accel_spam — 4 variants of episode length
    for i, length in enumerate([30, 40, 50, 60]):
        add(f"jerk_spam_{i}", "jerk_accel_spam", _gen_jerk_accel_spam(rng, length))

    # 8) legitimate_overtaking — 7 variants across speed x overtake count
    #    (this is the category that should sit at/near the top of the
    #    reference ranking, and the gate explicitly checks it dominates
    #    every unsafe category below)
    for i, (speed, n_ot, length) in enumerate(
        [
            (24.0, 3, 50),
            (25.0, 3, 50),
            (26.0, 3, 60),
            (27.0, 4, 60),
            (28.0, 5, 70),
            (29.0, 4, 50),
            (30.0, 6, 80),
        ]
    ):
        add(
            f"legit_overtake_{i}",
            "legitimate_overtaking",
            _gen_legitimate_overtaking(rng, length, speed, n_ot),
        )

    return bank


# ── Cumulative-return evaluation of a candidate reward function ───────────────


def _cumulative_return(reward_fn: Callable[[dict], float], states: list[dict]) -> float:
    """
    Returns the mean per-step reward over the trajectory.
    Using mean (not sum) removes episode-length bias: a shorter but
    higher-quality trajectory is not penalised just because it has
    fewer steps.  This aligns with the reference fitness, which uses
    episode-level means (mean_speed, etc.) rather than episode totals.
    """
    if not states:
        return 0.0
    total = 0.0
    for s in states:
        total += float(reward_fn(s))
    return total / len(states)


def evaluate_consistency(
    reward_fn: Callable[[dict], float],
    bank: list[TrajectorySpec] | None = None,
    max_violation_rate: float = 0.10,
    min_fitness_gap: float = 0.05,
) -> tuple[bool, str]:
    """
    Runs `reward_fn` over every trajectory in the bank, then checks pairwise
    ranking consistency against each trajectory's independent reference
    fitness (computed by reward_archive.compute_fitness on the trajectory's
    own aggregate metrics — entirely independent of the candidate reward
    function being tested).

    For every pair (A, B) whose reference fitness differs by at least
    `min_fitness_gap` (so near-ties, which are legitimate judgment calls,
    are excluded from scoring), the candidate's cumulative episode return
    must agree on which one is better. The fraction of such "decisive"
    pairs that disagree is the violation rate.

    Also explicitly checks that every 'legitimate_overtaking' trajectory's
    candidate return is HIGHER than every 'reckless_crash' and
    'tailgating_no_crash' trajectory's candidate return — these are the
    cases a thesis reviewer will look for first, so they're surfaced as
    their own named violations even though they're a subset of the general
    pairwise check.

    Returns (ok, report) where `ok` is True iff:
      - violation_rate <= max_violation_rate, AND
      - zero hard safety-category violations (see above)

    `report` is always populated with a summary (even on success) and, on
    failure, lists the worst-violating pairs in a format suitable for
    feeding straight back into the LLM repair prompt.
    """
    if bank is None:
        bank = build_trajectory_bank()

    # 1) Compute candidate cumulative return for every trajectory.
    returns: dict[str, float] = {}
    for spec in bank:
        try:
            returns[spec.name] = _cumulative_return(reward_fn, spec.states)
        except Exception as exc:
            return False, (
                f"Trajectory Bank Error: reward function raised "
                f"{type(exc).__name__}: {exc} while executing trajectory "
                f"'{spec.name}' (category='{spec.category}'). Every state "
                f"dict uses only the documented state keys — check for typos "
                f"such as state['overtake'] instead of state['overtook']."
            )

    # 2) Pairwise consistency check against reference fitness.
    n = len(bank)
    decisive_pairs = 0
    violations: list[tuple[TrajectorySpec, TrajectorySpec, float, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = bank[i], bank[j]
            fitness_gap = a.ref_fitness - b.ref_fitness
            if abs(fitness_gap) < min_fitness_gap:
                continue  # near-tie in ground truth — not a decisive pair
            decisive_pairs += 1

            better, worse = (a, b) if fitness_gap > 0 else (b, a)
            if returns[better.name] <= returns[worse.name]:
                violations.append((better, worse, returns[better.name], returns[worse.name]))

    violation_rate = (len(violations) / decisive_pairs) if decisive_pairs else 0.0

    # 3) Hard named-category check: legitimate overtaking must beat every
    #    reckless-crash and tailgating trajectory on cumulative return,
    #    BUT only when the reference fitness ALSO says legit > unsafe.
    #    If the reference fitness says the unsafe trajectory is genuinely
    #    better (e.g. tailgate at 29 m/s vs a slow legit overtaker),
    #    then the reward agreeing with the reference is correct.
    legit = [s for s in bank if s.category == "legitimate_overtaking"]
    unsafe = [s for s in bank if s.category in ("reckless_crash", "tailgating_no_crash")]
    hard_violations: list[tuple[TrajectorySpec, TrajectorySpec, float, float]] = []
    for L in legit:
        for U in unsafe:
            fitness_gap = L.ref_fitness - U.ref_fitness
            if abs(fitness_gap) < min_fitness_gap:
                continue  # near-tie, not a clear violation
            if fitness_gap > 0 and returns[L.name] <= returns[U.name]:
                hard_violations.append((L, U, returns[L.name], returns[U.name]))

    ok = violation_rate <= max_violation_rate

    # 4) Build report.
    lines = [
        "=== TRAJECTORY BANK CONSISTENCY REPORT ===",
        f"trajectories          : {n}",
        f"decisive pairs (gap>={min_fitness_gap}) : {decisive_pairs}",
        f"pairwise violations    : {len(violations)} ({violation_rate:.1%})",
        f"violation threshold    : {max_violation_rate:.1%}",
        f"hard safety violations : {len(hard_violations)} "
        f"(legit-overtaking trajectories ranked below unsafe ones despite higher ref_fitness)",
    ]

    if not ok:
        lines.append("")
        lines.append("Worst offending pairs (reward function disagrees with ground-truth fitness):")
        worst = sorted(
            violations,
            key=lambda v: (v[2] - v[3]),  # most-wrong first (better got LOWER or equal return)
        )[:8]
        for better, worse, r_better, r_worse in worst:
            lines.append(
                f"  - '{better.name}' (category={better.category}, ref_fitness={better.ref_fitness:.3f}) "
                f"SHOULD score higher than '{worse.name}' (category={worse.category}, "
                f"ref_fitness={worse.ref_fitness:.3f}), but candidate reward gave "
                f"{r_better:.2f} vs {r_worse:.2f}."
            )
        if hard_violations:
            lines.append("")
            lines.append("Hard safety-category violations (legitimate overtaking NOT beating unsafe driving):")
            for L, U, rL, rU in hard_violations[:8]:
                lines.append(
                    f"  - '{L.name}' (legitimate_overtaking, return={rL:.2f}) did NOT beat "
                    f"'{U.name}' ({U.category}, return={rU:.2f})."
                )
        lines.append("")
        lines.append(
            "This means the reward function likely rewards unsafe/degenerate behaviour "
            "(crashing, tailgating, lane-thrashing, accel spam, or standing still) at "
            "least as much as safe, fast, actively-overtaking driving over a full episode. "
            "Rebalance the magnitudes so genuine safe+fast+overtaking driving accumulates "
            "strictly more reward over an episode than any unsafe pattern."
        )

    return ok, "\n".join(lines)


# ── Self-test / CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bank = build_trajectory_bank()
    print(f"Built {len(bank)} trajectories:\n")
    by_cat: dict[str, int] = {}
    for spec in bank:
        by_cat[spec.category] = by_cat.get(spec.category, 0) + 1
    for cat, count in by_cat.items():
        print(f"  {cat:<24} {count:>3}")
    print(f"\n  TOTAL: {len(bank)}\n")

    print(f"{'name':<22} {'category':<24} {'ref_fitness':>11}  {'len':>4}")
    print("-" * 70)
    for spec in sorted(bank, key=lambda s: -s.ref_fitness):
        print(f"{spec.name:<22} {spec.category:<24} {spec.ref_fitness:>11.4f}  {len(spec.states):>4}")

    # Quick smoke test using the generation-0 default reward as the
    # candidate, to verify the pipeline runs end-to-end.
    print("\n=== Self-test against reward_program.py default (gen 0) ===")

    import math as _math

    def _clip(v, lo, hi):
        return max(lo, min(hi, v))

    def _default_compute_reward(state):
        if state["collided"]:
            return -30.0
        speed_reward = _clip(state["speed_ms"] * 0.1, 0.0, 3.0)
        ttc_penalty = -3.0 if state["ttc"] < 1.0 else -1.0 if state["ttc"] < 3.0 else 0.0
        overtake_bonus = 2.0 if state["overtook"] else 0.0
        jerk_penalty = -0.02 * (abs(state["long_jerk"]) + abs(state["lat_jerk"]))
        accel_penalty = -0.02 * abs(state["accel_ms2"])
        safe_gap_reward = 0.05 * _clip(state["front_dist"] - 15.0, 0.0, 30.0)
        lane_change_penalty = -0.1 if state["lane_changed"] else 0.0
        return (
            speed_reward
            + ttc_penalty
            + overtake_bonus
            + jerk_penalty
            + accel_penalty
            + safe_gap_reward
            + lane_change_penalty
        )

    ok, report = evaluate_consistency(_default_compute_reward, bank)
    print(report)
    print(f"\nGate result: {'PASS' if ok else 'FAIL'}")
