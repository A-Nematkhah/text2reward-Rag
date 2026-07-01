"""Gym wrapper: parse observations, run compute_reward, collect episode_stats."""

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
from txt2reward.reward.clip import clip_shaped_reward
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

# Physical de-normalisation (highway-env KinematicObservation, normalize=True).
_SPEED_SCALE = HIGHWAY_SPEED_SCALE
_LANE_WIDTH = 4.0
_DT = 1.0 / 5.0
_PRESENCE_TH = 0.5
_DIST_SCALE = HIGHWAY_DIST_SCALE
_DIST_MAX = 200.0

# Overtake tracking: nearest-neighbour match with per-step jump gates.
_TRACK_MAX_DX_JUMP = 8.0
_TRACK_MAX_VX_JUMP = 6.0
_OVERTAKE_LANE_RANGE = 1
_TRACK_MAX_MISSES = 3
_OVERTAKE_REARM_MARGIN = 2.0


def _clip_shaped_reward(reward: float, *, collided: bool = False) -> float:
    """Backward-compatible alias for clip_shaped_reward."""
    return clip_shaped_reward(reward, collided=collided)


def _denorm_y(y_raw: float, num_lanes: int, normalised: bool) -> float:
    """Lateral offset in metres (same formula for ego and other vehicles)."""
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
    """Executes compute_reward from reward_program.py; optional shaped reward."""

    MAX_TRAJ_SAMPLES = 8

    def __init__(
        self,
        env: gym.Env,
        reload_interval: int = DEFAULT_RELOAD_INTERVAL,
        num_lanes: int = 4,
        reward_path: str = REWARD_PROGRAM_PATH,
        reward_timeout_sec: float = REWARD_STEP_TIMEOUT_SEC,
        apply_shaped_reward: bool = True,
        weights_path: str | None = None,
        llm_interval: int = 50,
    ):
        super().__init__(env)
        self.reload_interval = reload_interval
        self.num_lanes = num_lanes
        self.reward_path = reward_path
        self.reward_timeout_sec = reward_timeout_sec
        self.apply_shaped_reward = apply_shaped_reward

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

        # Execute reward function in sandbox (skip when only collecting stats).
        shaped_reward = 0.0
        raw_reward: float | None = None
        if self.apply_shaped_reward:
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

            raw_reward = float(shaped_reward)
            shaped_reward = _clip_shaped_reward(shaped_reward, collided=collided)

        # Debug logging: raw vs clipped (collisions always; otherwise every 1000 steps)
        if self.apply_shaped_reward and os.environ.get("DEBUG_REWARD"):
            periodic = self._global_step % 1000 == 0
            if collided or periodic:
                if raw_reward is not None and abs(raw_reward - shaped_reward) > 1e-6:
                    reward_part = f"raw={raw_reward:.3f} -> clipped={shaped_reward:.3f}"
                else:
                    reward_part = f"reward={shaped_reward:.3f}"
                log.debug(
                    f"[wrapper] step={self._global_step:6d} "
                    f"speed={state['speed_ms']:.1f} m/s  "
                    f"front={state['front_dist']:.1f} m  "
                    f"ttc={state['ttc']:.1f} s  "
                    f"{reward_part}  "
                    f"collided={collided}  "
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

        step_reward = shaped_reward if self.apply_shaped_reward else float(env_reward)

        return obs, step_reward, terminated, truncated, info


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
