"""LLM prompt templates for reward generation and critique."""

from txt2reward.config.llm import LLM_MODEL

MODEL = LLM_MODEL

# Shipped bootstrap body (no header) — used when LLM bootstrap fails and disk is empty.
DEFAULT_BOOTSTRAP_REWARD_BODY = """\
def compute_reward(state):
    if state["collided"]:
        return -80.0
    speed = state["speed_ms"]
    open_road = state["front_dist"] > 35.0 and state["ttc"] > 5.0
    clear_road = state["front_dist"] > 40.0 and state["ttc"] > 5.0
    speed_reward = clip(speed * 0.09, 0.0, 3.0)
    slow_penalty = -0.5 if clear_road and speed < 22.0 else 0.0
    passive_band_tax = -0.65 if open_road and not state["overtook"] and speed > 18.0 and speed <= 22.0 else 0.0
    static_passive_tax = -0.35 if open_road and not state["overtook"] and not state["lane_changed"] else 0.0
    cruise_tax = -1.8 if open_road and not state["overtook"] and speed > 22.0 else 0.0
    no_overtake_tax = (
        -0.85
        if open_road and not state["overtook"] and not state["lane_changed"]
        else (-0.4 if open_road and not state["overtook"] else 0.0)
    )
    ttc_penalty = -4.0 if state["ttc"] < 1.0 else -2.0 if state["ttc"] < 3.0 else 0.0
    tailgate_penalty = -1.8 if state["front_dist"] < 20.0 and state["ttc"] < 4.0 else 0.0
    overtake_bonus = 3.0 if state["overtook"] else 0.0
    jerk_penalty = -0.45 * (abs(state["long_jerk"]) + abs(state["lat_jerk"]))
    accel_penalty = -0.20 * abs(state["accel_ms2"])
    gap_bonus = 0.003 * clip(state["front_dist"] - 25.0, 0.0, 10.0) if speed >= 22.0 else 0.0
    lc_penalty = -0.55 if state["lane_changed"] and not state["overtook"] else 0.0
    return (
        speed_reward
        + slow_penalty
        + passive_band_tax
        + static_passive_tax
        + cruise_tax
        + no_overtake_tax
        + ttc_penalty
        + tailgate_penalty
        + overtake_bonus
        + jerk_penalty
        + accel_penalty
        + gap_bonus
        + lc_penalty
    )
"""


_STATE_SCHEMA = """\
State keys available inside compute_reward(state):
  speed_ms        : float   ego speed in m/s (range 0-40)
  front_dist      : float   distance to front vehicle [m] (0-200, 200 = clear)
  ttc             : float   time-to-collision [s] (0-30, 30 = no threat)
  rel_vel_ms      : float   v_front - v_ego [m/s] (negative = approaching)
  lane            : int     lane index, 0 = rightmost
  overtook        : bool    completed an overtake this step   ← NOTE: "overtook", NOT "overtake"
  lane_changed    : bool    lane changed since last step
  collided        : bool    collision detected
  nearby_vehicles : int     vehicles within ~30 m radius
  accel_ms2       : float   longitudinal acceleration [m/s2]
  long_jerk       : float   longitudinal jerk [m/s3]
  lat_jerk        : float   lateral jerk [m/s3]

CRITICAL: The only valid state keys are EXACTLY the ones listed above.
  * Use state["overtook"]  ← correct (past tense, with k)
  * NEVER use state["overtake"]  ← this key does NOT exist and will crash

Safe math available (no imports, just use by name):
  min, max, abs, round, float, int, bool
  sqrt, exp, log, sin, cos, tan, atan, atan2
  floor, ceil, clip(val, lo, hi), pi, e, inf
"""

_GENERATION_SYSTEM = """\
You are a reinforcement learning reward engineer for a PPO highway driving agent.

Your task: write a Python function compute_reward(state) that returns a float.
The agent drives on a 4-lane highway. Goal: high-speed, safe, efficient driving with active overtaking.

HARD RULES (violation = sandbox rejection):
    * Function signature: def compute_reward(state):
    * HARD SAFETY CHECK: If state["collided"] is True, the function MUST immediately
      return the collision penalty (e.g., -80.0) without any other positive terms,
      speed rewards, or bonuses calculated or added in that same step. You MUST use
      exactly this pattern at the very beginning of the function body:
          if state["collided"]:
              return -80.0
    * No import statements
    * No attribute access (no obj.method)
    * No loops (for/while)
    * No builtins except: min, max, abs, round, float, int, bool
    * Only approved math: sqrt, exp, log, sin, cos, tan, atan2, floor, ceil, clip, pi
    * Must return a float value
    * Single local variables allowed; no nested functions

DESIGN PRINCIPLES:
    * Collision penalty MUST dominate (-60 to -100). A typical episode is ~40 steps;
      per-step speed reward must NOT make crashing net-profitable. Rule of thumb:
      40 steps × max per-step reward < |collision penalty|.
    * Speed reward: moderate coefficient (0.06–0.10), cap ≤ 3.0 — enough to prefer
      28 m/s over 14 m/s under safe conditions, but not so large it pays to crash.
    * TTC penalty should activate below 3 s and be strong enough to prevent tailgating.
    * Overtake bonus: one-shot (+2 to +4) when overtook == True
    * Jerk/accel penalties: 0.10–0.20 scale — must penalise jerk_accel_spam trajectories
    * Avoid large safe_gap / front_dist bonuses — passive cruising exploit
    * Lane change without overtake: penalise (-0.4 to -0.6)
    * ANTI-CRASH-FARMING: the validation pipeline simulates a 39-step fast drive plus
      a collision; that episodic total MUST be lower than a full safe cautious episode.
    * ANTI-PASSIVE-DRIVING: cruise_tax on clear road above 22 m/s without overtakes
    * SPEED INCENTIVE TEST: 28 m/s must beat 14 m/s at identical safe conditions
    * Fitness v8 ranks lower crash_rate higher even above 50% crash — but crashing
      every episode still scores poorly; survival requires actually reducing crashes.

{state_schema}

Reply ONLY with the Python source of compute_reward(state). No explanation, no markdown fences.
"""

_CRITIQUE_SYSTEM = """\
You are a reinforcement learning reward auditor. Analyse the reward function and metrics below.

Identify:
1. Reward hacking patterns:
   - Passive driving / "slow to survive": crash_rate near 0 but mean_speed < 22 m/s
     and mean_overtakes near 0 — agent crawls to avoid risk instead of driving well
   - Oscillatory lane changes: lane_changes >> overtakes (agent thrashing lanes for reward)
   - Acceleration spam: high mean_accel with low speed gain (braking-acceleration exploit)
   - Stationary farming: very low mean_speed but high shaped_reward
   - TTC exploitation: very low ttc or min_ttc but no crashes (agent riding tailgate for some bonus)
2. Missing incentives (what good behaviour is not rewarded)
3. Misaligned incentives (what bad behaviour is inadvertently rewarded)
4. Proposed improvements with SPECIFIC code changes

Be concise (max 300 words). End with 3 concrete bullet-point improvements.

IMPORTANT: At the very end of your response, append a machine-readable metadata block
on a single line in this EXACT format (no whitespace before the colon):
CRITIQUE_META:{"failure_modes":["tag1","tag2"],"strengths":["s1"],"summary":"one sentence"}

Valid failure_mode tags: tailgating, passive_driving, oscillatory_lane_changes,
acceleration_spam, stationary_farming, reward_hacking
Valid strength tags: high_speed, good_overtaking, safe_driving, smooth_driving
"""

_GENERATION_USER_TEMPLATE = """\
=== DRIVING GOAL ===
{goal}

=== CURRICULUM PHASE: {curriculum_phase} ===
{curriculum_guidance}

=== ARCHIVE MEMORY ===
{archive_context}

The archive is organised into sections:
  A) Top performers — adopt their strengths
  B) Most recent — continue or fix the current trajectory
  C) Failed rewards — do NOT repeat their mistakes
  D) Similar failure modes — study why the same issues appeared before

=== TASK ===
Generate an improved compute_reward(state) function that achieves the goal above.
- Learn from top performers: adopt what scored well.
- Avoid the failure patterns shown in sections C and D.
- If a failure mode is listed (e.g. tailgating, passive_driving), explicitly add
  a term that penalises it.
- Do NOT replicate "safe but slow" rewards (0% crash, speed ~20 m/s, no overtakes).
  The fitness function now penalises this via a passive-driving gate.
- Prioritise: (1) no collisions, (2) speed >= 24 m/s when road is clear, (3) active overtaking.
Return ONLY the Python function source. No explanation, no markdown.
"""

_CRITIQUE_USER_TEMPLATE = """\
=== REWARD PROGRAM (Generation {generation}) ===
```python
{reward_code}
```

=== EVALUATION METRICS ===
  mean_speed       : {mean_speed:.2f} m/s
  crash_rate       : {crash_rate:.1%}
  mean_overtakes   : {mean_overtakes:.2f} per episode
  completion_rate  : {completion_rate:.1%}
  mean_steps       : {mean_steps:.0f}
  mean_ttc         : {mean_ttc:.2f} s
  p10_ttc          : {p10_ttc:.2f} s   (10th-percentile TTC — near-miss indicator)
  min_ttc          : {min_ttc:.2f} s   (worst single-step TTC)
  near_miss_rate   : {near_miss_rate:.1%} (fraction of steps with TTC < 2 s)
  safe_ot_ratio    : {safe_overtake_ratio:.2f} (overtakes / lane changes)
  lane_change_rate : {lane_change_rate:.2f} per episode
  curriculum_phase : {curriculum_phase}
  mean_long_jerk   : {mean_long_jerk:.3f} m/s3
  mean_accel       : {mean_accel:.3f} m/s2
  total_lc         : {total_lane_changes} lane changes
  fitness          : {fitness:.4f}

=== TREND VS PREVIOUS GENERATION ===
{trend_summary}

=== EPISODE TRAJECTORY SAMPLES ===
{trajectory_summary}

Identify reward hacking, failure modes, and propose 3 specific improvements.
If mean_speed or mean_overtakes is DECREASING while crash_rate also decreases,
treat this as a strong reward-hacking signal (the agent is likely slowing down
or refusing to overtake just to avoid crashing, instead of driving well) and
say so explicitly.
"""

_REPAIR_USER_TEMPLATE = """\
The reward function you generated failed validation with this error:

  {error}

Common causes:
  * Using a state key that does not exist, e.g. state["overtake"] — the correct
    key is state["overtook"] (past tense, with k). Other valid keys are:
    speed_ms, front_dist, ttc, rel_vel_ms, lane, overtook, lane_changed,
    collided, nearby_vehicles, accel_ms2, long_jerk, lat_jerk.
    DO NOT invent new key names.
  * Using a disallowed builtin or math function
  * Syntax errors, loops, or import statements

Here is the rejected code:
```python
{rejected_code}
```

Fix ALL issues and return ONLY the corrected compute_reward(state) function.
No explanation, no markdown fences.
"""
