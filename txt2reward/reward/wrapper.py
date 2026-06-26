"""
reward_wrapper.py
─────────────────
Gym wrapper that:
  1. Parses the full KinematicObservation on every step to extract state signals.
  2. Computes the shaped reward using the dynamically-loaded reward_program.py.
  3. Reloads reward_program.py every `reload_interval` steps.
  4. When an episode ends, stores a rich summary in info["episode_stats"].

Key change from weight-based system
────────────────────────────────────
  OLD: compute_shaped_reward(weights, speed_ms, lane, ...) — fixed formula
  NEW: compute_reward(state)                               — fully dynamic

The wrapper imports the generated reward function from reward_program.py using
importlib so hot-swapping works without restarting the training process.

SubprocVecEnv note
──────────────────
Each worker is a separate process. reward_program.py on disk is the shared
state. The wrapper reloads the module every `reload_interval` steps.

Observation layout (highway-v0, KinematicObservation, normalize=True):
  Each row → [presence, x, y, vx, vy]
  Row 0    → ego vehicle
  Rows 1…N → surrounding vehicles

State signals computed
──────────────────────
  speed_ms        ego speed [m/s]
  front_dist      distance to front vehicle [m]
  ttc             time-to-collision [s]
  rel_vel_ms      v_front - v_ego [m/s]
  lane            lane index
  lane_changed    True if lane changed since last step
  overtook        True if ego passed a trailing vehicle
  accel_ms2       longitudinal acceleration [m/s²]
  long_jerk       longitudinal jerk [m/s³]
  lat_jerk        lateral jerk [m/s³]
  nearby_vehicles count within density_radius metres
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from txt2reward.config.paths import REWARD_PROGRAM_PATH
from txt2reward.config.training import DEFAULT_RELOAD_INTERVAL
from txt2reward.config.validation import REWARD_STEP_TIMEOUT_SEC
from txt2reward.core.constants import HIGHWAY_DIST_SCALE, HIGHWAY_SPEED_SCALE
from txt2reward.core.log import get_logger
from txt2reward.core.metrics import percentile
from txt2reward.sandbox.sandbox import (
    build_state,
    execute_reward,
    extract_reward_body,
    validate_reward_code,
)

log = get_logger("wrapper")

# ── Observation column indices ────────────────────────────────────────────────
_IDX_PRESENCE = 0
_IDX_X = 1
_IDX_Y = 2
_IDX_VX = 3
_IDX_VY = 4

# ── Physical constants ────────────────────────────────────────────────────────
# Vehicle.MAX_SPEED = 40.0 m/s (see highway_env.envs.common.observation.KinematicObservation).
# So the correct de-normalisation factors are 2*40=80 for speed and 5*40=200 for
# distance — NOT 40 and 100. (Previously these were halved, which silently
# capped every speed/distance signal at half its true value.)
_SPEED_SCALE = HIGHWAY_SPEED_SCALE
_LANE_WIDTH = 4.0
_DT = 1.0 / 5.0
_PRESENCE_TH = 0.5
_DIST_SCALE = HIGHWAY_DIST_SCALE
_DIST_MAX = 200.0

# ── Overtake-tracking constants ───────────────────────────────────────────────
# Vehicles are tracked across steps by nearest-neighbour matching in
# (relative longitudinal distance, relative speed) space, gated by maximum
# plausible per-step jumps in each dimension. At policy_frequency=5 Hz a real
# vehicle cannot teleport, so a match candidate whose dx or vx changed by more
# than these bounds in one step is treated as a *different* vehicle rather
# than the same one re-detected — this is what prevents identity swaps in
# dense traffic (up to ~30 vehicles, ~10 visible in the observation).
#   _TRACK_MAX_DX_JUMP : at 5 Hz, even a large relative speed (~40 m/s closing)
#                        only covers 8 m per step, so an 8 m jump comfortably
#                        bounds normal motion/IDM jitter while still rejecting
#                        a match against a different, similarly-positioned car.
#   _TRACK_MAX_VX_JUMP : relative speed cannot swing by more than a few m/s in
#                        1/5 s under normal driving dynamics (no instantaneous
#                        speed changes), so 6 m/s safely bounds real jitter
#                        while rejecting a mismatched vehicle with a very
#                        different relative speed.
_TRACK_MAX_DX_JUMP = 8.0  # metres
_TRACK_MAX_VX_JUMP = 6.0  # m/s
# Only vehicles in the ego's lane or an immediately adjacent lane are
# candidates for "being overtaken" — a car three lanes over is not something
# the ego is meaningfully passing, even if it is technically ahead in x.
_OVERTAKE_LANE_RANGE = 1
# A tracked vehicle that hasn't been seen (matched) for this many consecutive
# steps is dropped from tracking, so stale tracks don't linger and falsely
# match a much-later, unrelated detection at a similar (dx, vx).
_TRACK_MAX_MISSES = 3
# Hysteresis margin for re-arming a track's overtake flag. A vehicle must
# clear back to dx > _OVERTAKE_REARM_MARGIN (not just dx > 0.0) before it is
# considered to have genuinely returned ahead of the ego. Without this
# margin, ordinary jitter for a vehicle riding alongside ego near dx=0
# (e.g. a same-speed neighbour in an adjacent lane) crosses the dx<=0
# boundary repeatedly, re-arming and re-firing a "new" overtake on every
# crossing even though no real pass is occurring. The margin must exceed
# plausible single-step jitter but stay well inside _TRACK_MAX_DX_JUMP so it
# doesn't interfere with genuine re-merge-ahead detection.
_OVERTAKE_REARM_MARGIN = 2.0  # metres


# ── Shared y/lateral de-normalization (SINGLE source of truth) ───────────────
#
# highway-env normalises the "y" feature into [-1, 1] using the range
# [-LANE_WIDTH * num_lanes, LANE_WIDTH * num_lanes] (see
# highway_env.envs.common.observation.KinematicObservation.normalize_obs,
# which derives the range from AbstractLane.DEFAULT_WIDTH * len(side_lanes),
# i.e. LANE_WIDTH * num_lanes for a one-way road). This is exactly the same
# kind of range-based normalisation already documented above for vx/x
# (_SPEED_SCALE / _DIST_SCALE).
#
# SECURITY/CORRECTNESS NOTE (fixes audit finding #1): there used to be TWO
# different, independently-derived multipliers applied to the same raw y
# value:
#   - the ego's lane index used  y_raw * num_lanes            (factor 4)
#   - another vehicle's dy_m used y_raw * LANE_WIDTH*(num_lanes-1)  (factor 12)
# Both were derived from the SAME underlying normalised y feature, so using
# two different factors meant the ego's own lateral position and other
# vehicles' lateral offsets were silently measured on two different scales.
# Every downstream signal that depends on lateral alignment (front_dist,
# ttc, nearby_vehicles, and the overtake lane-window gate) inherited this
# inconsistency. The fix is to de-normalise y ONCE, the same way, for both
# the ego and every other vehicle, and derive the lane index from that
# single metres value.
def _denorm_y(y_raw: float, num_lanes: int, normalised: bool) -> float:
    """Converts a raw 'y' observation value into metres of lateral offset.

    Used for BOTH the ego's absolute lateral position and other vehicles'
    relative dy -- they must use the identical formula since they come from
    the same normalisation range.
    """
    return y_raw * _LANE_WIDTH * num_lanes if normalised else y_raw


def _lane_from_y_m(y_m: float, num_lanes: int) -> int:
    """Converts a lateral offset in metres to a clipped lane index.

    Lane centres sit at y = 0, LANE_WIDTH, 2*LANE_WIDTH, ... so the lane
    index is simply the offset divided by LANE_WIDTH, rounded and clipped
    to the valid lane range.
    """
    return int(np.clip(round(y_m / _LANE_WIDTH), 0, num_lanes - 1))


# path -> (mtime, compute_reward callable) — skip disk/exec when file unchanged
_REWARD_FN_CACHE: dict[str, tuple[float, Callable]] = {}


def clear_reward_fn_cache(path: str | None = None) -> None:
    """Drop cached reward loaders after disk writes (security + correctness)."""
    if path is None:
        _REWARD_FN_CACHE.clear()
    else:
        _REWARD_FN_CACHE.pop(path, None)


def _load_reward_fn(path: str, *, validate: bool = True):
    """
    Dynamically loads compute_reward() from reward_program.py.

    Injects the same safe math namespace used by the sandbox (clip, sqrt,
    exp, etc.) so generated code that relies on these helpers works whether
    executed directly here or validated/executed via reward_sandbox.

    Falls back to a simple speed reward if the file is missing or invalid.
    """
    if not os.path.exists(path):
        return _fallback_reward

    try:
        mtime = os.path.getmtime(path)
        cached = _REWARD_FN_CACHE.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        with open(path, encoding="utf-8") as f:
            source = f.read()
        body = extract_reward_body(source)
        if validate:
            ok, err = validate_reward_code(body)
            if not ok:
                log.warning("[wrapper] %s failed AST validation on reload: %s — using fallback", path, err)
                return _fallback_reward

        from txt2reward.sandbox.sandbox import _make_safe_namespace

        spec = importlib.util.spec_from_file_location("reward_program", path)
        mod = importlib.util.module_from_spec(spec)

        # Inject safe math helpers (clip, sqrt, exp, ...) before exec
        safe_ns = _make_safe_namespace()
        safe_ns.pop("__builtins__", None)
        for k, v in safe_ns.items():
            setattr(mod, k, v)

        # SECURITY (fixes audit finding #6): explicitly strip real Python
        # builtins from the module namespace before exec_module() runs.
        # importlib's exec_module() calls exec() under the hood, and exec()
        # auto-injects the REAL __builtins__ dict (open, eval, exec,
        # __import__, ...) into the globals it's given if one isn't already
        # present. The line above only sets approved helper names as plain
        # module attributes -- it never sets mod.__builtins__ itself, so
        # without this explicit assignment the generated code would still
        # have full access to real builtins at runtime. AST validation
        # (reward_sandbox.validate_reward_code) is the primary defense and
        # already forbids names like `eval`/`exec`/`__import__`, but this is
        # a deliberate second line of defense: if a reward program ever
        # reaches this loader without having been (re-)validated -- e.g. a
        # corrupted or pre-validation legacy archive entry -- real builtins
        # must still not be reachable from inside compute_reward().
        mod.__dict__["__builtins__"] = {}

        spec.loader.exec_module(mod)
        fn = getattr(mod, "compute_reward", None)
        if fn is None:
            log.warning(f"[wrapper] compute_reward not found in {path} — using fallback")
            return _fallback_reward
        _REWARD_FN_CACHE[path] = (mtime, fn)
        return fn
    except Exception as e:
        log.warning(f"[wrapper] Failed to load {path}: {e} — using fallback")
        return _fallback_reward


def _fallback_reward(state: dict) -> float:
    """Emergency fallback when reward_program.py is unavailable."""
    speed_norm = min(1.0, state.get("speed_ms", 0.0) / 30.0)
    collision = -20.0 if state.get("collided", False) else 0.0
    return 0.8 * speed_norm + collision


class LLMRewardWrapper(gym.Wrapper):
    """
    Gym wrapper that executes the dynamically generated reward function.

    Backward-compatible with train.py: the constructor signature is unchanged
    (except `weights_path` is replaced by `reward_path`, with a default that
    matches the old WEIGHTS_FILE name for easy migration).
    """

    MAX_TRAJ_SAMPLES = 8

    def __init__(
        self,
        env: gym.Env,
        reload_interval: int = DEFAULT_RELOAD_INTERVAL,
        num_lanes: int = 4,
        reward_path: str = REWARD_PROGRAM_PATH,
        reward_timeout_sec: float = REWARD_STEP_TIMEOUT_SEC,
        # backward-compat stubs
        weights_path: str | None = None,
        llm_interval: int = 50,
    ):
        super().__init__(env)
        self.reload_interval = reload_interval
        self.num_lanes = num_lanes
        self.reward_path = reward_path
        self.reward_timeout_sec = reward_timeout_sec

        self._reward_fn = _load_reward_fn(self.reward_path)
        self._global_step = 0
        self._density_radius = 30.0  # metres; fixed (no weight dict anymore)

        self._reset_episode_accum()

    # ── Episode accumulators ──────────────────────────────────────────────────

    def _reset_episode_accum(self) -> None:
        self._ep_env_reward = 0.0
        self._ep_shaped_reward = 0.0
        self._ep_speed_sum = 0.0
        self._ep_dist_sum = 0.0
        self._ep_steps = 0
        self._ep_collisions = 0

        self._ep_ttc_sum = 0.0
        self._ep_ttc_vals: list[float] = []
        self._ep_rel_vel_sum = 0.0
        self._ep_long_jerk_sum = 0.0
        self._ep_lat_jerk_sum = 0.0
        self._ep_accel_sum = 0.0
        self._ep_density_sum = 0.0
        self._ep_overtakes = 0
        self._ep_lane_changes = 0

        self._ep_traj: list[dict] = []

        self._prev_speed_ms: float | None = None
        self._prev_accel_ms2: float = 0.0
        self._prev_lat_vel_ms: float = 0.0
        self._prev_lane: int | None = None
        # Persistent nearest-neighbour vehicle tracker for overtake detection.
        # See _match_tracks() / _update_overtake_tracks() for the tracking
        # scheme. Each track is a dict: {dx, vx, misses, overtaken}.
        self._overtake_tracks: list[dict] = []
        self._next_track_id: int = 0

    def reset(self, **kwargs):
        self._reset_episode_accum()
        return self.env.reset(**kwargs)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._global_step += 1

        # Reload reward function periodically
        if self._global_step % self.reload_interval == 0:
            self._reward_fn = _load_reward_fn(self.reward_path)

        # Parse state from observation
        parsed = _parse_full_obs(
            obs,
            num_lanes=self.num_lanes,
            prev_speed_ms=self._prev_speed_ms,
            prev_accel_ms2=self._prev_accel_ms2,
            prev_lat_vel_ms=self._prev_lat_vel_ms,
            prev_lane=self._prev_lane,
            overtake_tracks=self._overtake_tracks,
            next_track_id=self._next_track_id,
            density_radius_m=self._density_radius,
        )

        collided = bool(info.get("crashed", False))

        # Build canonical state dict
        state = build_state(parsed, collided)

        # Execute reward function in sandbox.
        #
        # SECURITY (fixes audit finding #2): this now actually goes through
        # reward_sandbox.execute_reward(), which runs the call inside a
        # thread with a hard timeout (self.reward_timeout_sec). Previously
        # this called execute_reward.__wrapped__ -- a monkey-patched
        # attribute that pointed at a plain, un-timeboxed direct call -- so
        # the real sandboxed/timeout-protected execute_reward() was never
        # actually invoked on the hot path, and an LLM-generated reward
        # program with a runaway computation (e.g. a pathological exponent)
        # could hang a worker process indefinitely.
        #
        # `compiled_fn=self._reward_fn` reuses the already-loaded function
        # (loaded once per reload_interval steps by _load_reward_fn) so we
        # still avoid recompiling the source on every single environment
        # step -- only the function CALL itself is timeboxed.
        try:
            shaped_reward = execute_reward(
                code="",
                state=state,
                timeout_sec=self.reward_timeout_sec,
                compiled_fn=self._reward_fn,
            )
        except Exception as e:
            # Fallback if execution fails or times out
            if self._global_step % 1000 == 1:
                log.warning(f"[wrapper] Reward execution error: {e}")
            shaped_reward = _fallback_reward(state)

        # Debug logging
        if os.environ.get("DEBUG_REWARD") and self._global_step % 1000 == 0:
            log.debug(
                f"[wrapper] step={self._global_step:6d} "
                f"speed={state['speed_ms']:.1f} m/s  "
                f"front={state['front_dist']:.1f} m  "
                f"ttc={state['ttc']:.1f} s  "
                f"reward={shaped_reward:.3f}  "
                f"overtook={state['overtook']}"
            )

        # Carry state forward
        self._prev_speed_ms = parsed["speed_ms"]
        self._prev_accel_ms2 = parsed["accel_ms2"]
        self._prev_lat_vel_ms = parsed["lat_vel_ms"]
        self._prev_lane = parsed["lane"]
        self._overtake_tracks = parsed["overtake_tracks"]
        self._next_track_id = parsed["next_track_id"]

        # Accumulate episode statistics
        self._ep_env_reward += env_reward
        self._ep_shaped_reward += shaped_reward
        self._ep_speed_sum += state["speed_ms"]
        self._ep_dist_sum += state["front_dist"]
        self._ep_steps += 1

        if collided:
            self._ep_collisions += 1

        self._ep_ttc_sum += state["ttc"]
        self._ep_ttc_vals.append(state["ttc"])
        self._ep_rel_vel_sum += state["rel_vel_ms"]
        self._ep_long_jerk_sum += abs(state["long_jerk"])
        self._ep_lat_jerk_sum += abs(state["lat_jerk"])
        self._ep_accel_sum += abs(state["accel_ms2"])
        self._ep_density_sum += state["nearby_vehicles"]

        if state["overtook"]:
            self._ep_overtakes += 1
        if state["lane_changed"]:
            self._ep_lane_changes += 1

        # Trajectory sample
        sample_every = max(1, 40 // self.MAX_TRAJ_SAMPLES)
        if self._ep_steps % sample_every == 0 and len(self._ep_traj) < self.MAX_TRAJ_SAMPLES:
            self._ep_traj.append(
                {
                    "speed_ms": round(state["speed_ms"], 2),
                    "lane": state["lane"],
                    "front_dist": round(state["front_dist"], 1),
                    "collided": collided,
                    "ttc": round(state["ttc"], 1),
                    "rel_vel_ms": round(state["rel_vel_ms"], 2),
                    "accel_ms2": round(state["accel_ms2"], 2),
                    "nearby_vehicles": state["nearby_vehicles"],
                    "overtook": state["overtook"],
                }
            )

        # Episode summary
        if terminated or truncated:
            n = max(self._ep_steps, 1)
            info = dict(info)
            info["episode_stats"] = {
                "total_env_reward": round(self._ep_env_reward, 3),
                "total_shaped_reward": round(self._ep_shaped_reward, 3),
                "mean_speed": round(self._ep_speed_sum / n, 2),
                "mean_front_dist": round(self._ep_dist_sum / n, 2),
                "collisions": self._ep_collisions,
                "steps": self._ep_steps,
                "mean_ttc": round(self._ep_ttc_sum / n, 2),
                "min_ttc": round(min(self._ep_ttc_vals) if self._ep_ttc_vals else 30.0, 2),
                "p10_ttc": round(percentile(self._ep_ttc_vals, 10), 2),
                "ttc_vals": list(self._ep_ttc_vals),
                "mean_rel_vel": round(self._ep_rel_vel_sum / n, 3),
                "mean_long_jerk": round(self._ep_long_jerk_sum / n, 3),
                "mean_lat_jerk": round(self._ep_lat_jerk_sum / n, 3),
                "mean_accel": round(self._ep_accel_sum / n, 3),
                "mean_density": round(self._ep_density_sum / n, 2),
                "total_overtakes": self._ep_overtakes,
                "total_lane_changes": self._ep_lane_changes,
                "trajectory_samples": list(self._ep_traj),
            }

        return obs, shaped_reward, terminated, truncated, info


# NOTE (fixes audit finding #2): there used to be a monkey-patch here
# (`execute_reward.__wrapped__ = _direct_execute`) that silently replaced
# the timeout-protected execute_reward() with a plain, un-timeboxed direct
# call, so the per-step hot path above never actually went through the
# sandbox's timeout guard. That patch and its `_direct_execute` helper have
# been removed -- the step() method now calls reward_sandbox.execute_reward()
# directly (with compiled_fn=self._reward_fn so the source isn't recompiled
# every step), which is the single, real execution path with a hard timeout.


# ── Overtake tracking: nearest-neighbour vehicle identity across steps ────────


def _match_track(
    tracks: list[dict],
    dx_m: float,
    vx_rel_ms: float,
    used: set[int],
) -> int | None:
    """
    Finds the best matching existing track for a new detection at
    (dx_m, vx_rel_ms), gated by the maximum plausible per-step jump in each
    dimension (_TRACK_MAX_DX_JUMP, _TRACK_MAX_VX_JUMP).

    Among all tracks within both gates, picks the nearest in normalised
    (dx, vx) space — this is the "nearest neighbour, gated by plausibility"
    matching scheme: a vehicle can't have moved further than physically
    possible in 1/5 s, so any track outside the gate is necessarily a
    different car and is excluded from the candidate set entirely, rather
    than merely being penalised in the distance metric.

    `used` holds indices already claimed by another detection this step, so
    two different new detections cannot both match the same stale track.

    Returns the matching track's index in `tracks`, or None if no track is
    within the gate (i.e. this is a newly-appeared vehicle).
    """
    best_idx: int | None = None
    best_score = float("inf")

    for idx, tr in enumerate(tracks):
        if idx in used:
            continue
        ddx = abs(dx_m - tr["dx"])
        dvx = abs(vx_rel_ms - tr["vx"])
        if ddx > _TRACK_MAX_DX_JUMP or dvx > _TRACK_MAX_VX_JUMP:
            continue
        # Normalised combined distance so dx (metres) and vx (m/s) contribute
        # comparably to the nearest-neighbour score.
        score = (ddx / _TRACK_MAX_DX_JUMP) ** 2 + (dvx / _TRACK_MAX_VX_JUMP) ** 2
        if score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def _update_overtake_tracks(
    tracks: list[dict],
    next_track_id: int,
    detections: list[tuple[float, float]],
) -> tuple[list[dict], int, bool]:
    """
    Advances the persistent vehicle tracker by one step and detects overtakes.

    Parameters
    ──────────
    tracks         : tracks from the previous step, each a dict with keys
                      {id, dx, vx, misses, overtaken}. `dx` is the relative
                      longitudinal distance to ego (positive = ahead),
                      `vx` is the relative speed, `overtaken` marks a track
                      that has already fired its one-shot overtake event
                      since it was last ahead of the ego.
    next_track_id  : monotonically increasing counter for new track IDs.
    detections     : this step's qualifying (same-lane / adjacent-lane)
                      vehicle detections as (dx_m, vx_rel_ms) pairs.

    Returns
    ───────
    (new_tracks, new_next_track_id, overtook)

    Matching is nearest-neighbour gated by maximum plausible per-step jumps
    (_match_track), so the same physical vehicle keeps the same track id
    across steps despite ordinary IDM jitter, while two physically distinct
    vehicles in dense traffic are not merged into one track.

    An overtake fires (exactly once per real pass) when a track's dx
    transitions from > 0 (ahead) to <= 0 (behind/level) between consecutive
    matched steps, gated by `overtaken` so a vehicle sitting behind the ego
    across many subsequent steps doesn't keep re-firing. The gate only
    resets (re-arms) once the vehicle clears back to dx > _OVERTAKE_REARM_MARGIN
    — not merely dx > 0.0 — which allows a legitimate double-overtake
    (re-merge ahead, then get passed again) while preventing ordinary
    jitter around dx=0 (e.g. a same-speed neighbour riding alongside the
    ego in an adjacent lane) from re-arming and re-firing on every small
    crossing of the zero boundary.
    """
    used: set[int] = set()
    new_tracks: list[dict] = []
    overtook = False

    for dx_m, vx_rel_ms in detections:
        match_idx = _match_track(tracks, dx_m, vx_rel_ms, used)

        if match_idx is None:
            # Newly appeared vehicle — start a fresh track. No overtake can
            # fire on a track's first sighting since there is no prior dx to
            # compare against (and a vehicle entering already-behind is not
            # an observed passing event, just a vehicle becoming visible).
            new_tracks.append(
                {
                    "id": next_track_id,
                    "dx": dx_m,
                    "vx": vx_rel_ms,
                    "misses": 0,
                    "overtaken": dx_m <= 0.0,
                }
            )
            next_track_id += 1
            continue

        used.add(match_idx)
        prev = tracks[match_idx]

        # Sign-change detection: ahead (>0) last step, behind/level (<=0) now,
        # and not already counted for this pass (prev["overtaken"] False).
        if prev["dx"] > 0.0 and dx_m <= 0.0 and not prev["overtaken"]:
            overtook = True
            new_overtaken = True
        elif dx_m > _OVERTAKE_REARM_MARGIN:
            # Vehicle has genuinely cleared back ahead of the ego (beyond the
            # hysteresis margin, not just a small jitter blip across dx=0) —
            # re-arm so a real future re-pass (e.g. it re-merges ahead after
            # a lane change) can fire again.
            new_overtaken = False
        else:
            # Still behind/level, or ahead but within the jitter margin of
            # dx=0, or already counted — stay armed-off so ordinary noise
            # near the crossing point doesn't recount the same pass.
            new_overtaken = prev["overtaken"]

        new_tracks.append(
            {
                "id": prev["id"],
                "dx": dx_m,
                "vx": vx_rel_ms,
                "misses": 0,
                "overtaken": new_overtaken,
            }
        )

    # Carry forward unmatched tracks (vehicle temporarily out of the
    # qualifying lane window or briefly undetected) up to _TRACK_MAX_MISSES
    # steps, so a one-frame dropout doesn't fragment a track's identity and
    # spuriously re-fire an overtake. Beyond that, the track is dropped.
    for idx, tr in enumerate(tracks):
        if idx in used:
            continue
        misses = tr["misses"] + 1
        if misses > _TRACK_MAX_MISSES:
            continue
        carried = dict(tr)
        carried["misses"] = misses
        new_tracks.append(carried)

    return new_tracks, next_track_id, overtook


# ── Full observation parser ───────────────────────────────────────────────────


def _parse_full_obs(
    obs: np.ndarray,
    num_lanes: int,
    prev_speed_ms: float | None,
    prev_accel_ms2: float,
    prev_lat_vel_ms: float,
    prev_lane: int | None,
    overtake_tracks: list[dict],
    next_track_id: int,
    density_radius_m: float,
) -> dict:
    """Parses KinematicObservation into state signals."""
    ego = obs[0]

    vx_raw = float(ego[_IDX_VX])
    normalised = abs(vx_raw) <= 1.5

    speed_ms = vx_raw * _SPEED_SCALE if normalised else vx_raw
    speed_ms = max(0.0, speed_ms)

    lat_vel_ms = float(ego[_IDX_VY]) * (_SPEED_SCALE if normalised else 1.0)

    y_raw = float(ego[_IDX_Y])
    y_m = _denorm_y(y_raw, num_lanes, normalised)
    lane = _lane_from_y_m(y_m, num_lanes)

    lane_changed = (prev_lane is not None) and (lane != prev_lane)

    front_dist = _DIST_MAX
    front_vx_ms = speed_ms
    nearby_count = 0

    # Candidate detections for overtake tracking: only vehicles in the ego's
    # lane or an immediately adjacent lane (_OVERTAKE_LANE_RANGE) are
    # relevant to "being overtaken" — a car several lanes over is excluded
    # even though it may be technically ahead in x.
    overtake_detections: list[tuple[float, float]] = []

    for i in range(1, len(obs)):
        row = obs[i]
        if float(row[_IDX_PRESENCE]) < _PRESENCE_TH:
            continue

        veh_x_raw = float(row[_IDX_X])
        dx_m = (veh_x_raw * _DIST_SCALE) if normalised else veh_x_raw

        veh_vx = float(row[_IDX_VX])
        veh_vx_ms = veh_vx * _SPEED_SCALE if normalised else veh_vx

        veh_y_raw = float(row[_IDX_Y])
        dy_m = _denorm_y(veh_y_raw, num_lanes, normalised)

        if 0.0 < dx_m < front_dist and abs(dy_m) < _LANE_WIDTH * 1.5:
            front_dist = dx_m
            front_vx_ms = veh_vx_ms

        if abs(dx_m) < density_radius_m and abs(dy_m) < _LANE_WIDTH * 1.5:
            nearby_count += 1

        # Lane-window gate for overtake tracking: same lane or adjacent lane
        # only (within _OVERTAKE_LANE_RANGE lane-widths of the ego's y).
        if abs(dy_m) <= _LANE_WIDTH * (_OVERTAKE_LANE_RANGE + 0.5):
            vx_rel_ms = veh_vx_ms - speed_ms
            overtake_detections.append((dx_m, vx_rel_ms))

    front_dist = float(np.clip(front_dist, 0.0, _DIST_MAX))
    rel_vel_ms = front_vx_ms - speed_ms

    if front_dist >= _DIST_MAX:
        ttc = 30.0
    else:
        closing_speed = max(-rel_vel_ms, 1e-6)
        ttc = front_dist / closing_speed
    ttc = float(np.clip(ttc, 0.0, 30.0))

    accel_ms2 = 0.0 if prev_speed_ms is None else (speed_ms - prev_speed_ms) / _DT
    long_jerk = (accel_ms2 - prev_accel_ms2) / _DT
    lat_jerk = (lat_vel_ms - prev_lat_vel_ms) / _DT

    new_tracks, new_next_track_id, overtook = _update_overtake_tracks(
        overtake_tracks, next_track_id, overtake_detections
    )

    return {
        "speed_ms": speed_ms,
        "lat_vel_ms": lat_vel_ms,
        "overtake_tracks": new_tracks,
        "next_track_id": new_next_track_id,
        "lane": lane,
        "front_dist": front_dist,
        "rel_vel_ms": rel_vel_ms,
        "ttc": ttc,
        "accel_ms2": accel_ms2,
        "long_jerk": long_jerk,
        "lat_jerk": lat_jerk,
        "nearby_vehicles": nearby_count,
        "overtook": overtook,
        "lane_changed": lane_changed,
    }
