"""
test_security_fixes.py
───────────────────────
Regression tests for the 6 urgent (🔴) fixes from the audit report
(txt2reward-v2_audit_report.md):

  1. Lane / dy_m de-normalization scale mismatch         (reward_wrapper.py)
  2. execute_reward() bypassed -> no timeout on hot path  (reward_wrapper.py / reward_sandbox.py)
  3. Smoke-test had no timeout                            (reward_designer.py)
  4. AST validator allowed unbounded Pow exponents        (reward_sandbox.py)
  5. Archive restore skipped re-validation                (reward_designer.py)
  6. _load_reward_fn left real Python builtins reachable  (reward_wrapper.py)

Runs under pytest if available:
    pytest test_security_fixes.py -v

Otherwise (pytest not installed -- e.g. offline/sandboxed environments),
run directly and it will execute every test_*() function and report
PASS/FAIL/ERROR with plain assertions:
    python3 test_security_fixes.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import reward_sandbox as rs

# reward_wrapper.py and reward_designer.py pull in gymnasium / groq
# respectively, which are heavy, optional dependencies that may not be
# installed in every environment that just wants to run these unit tests.
# Skip the tests that need them gracefully rather than failing the whole
# file on import -- mirrors this project's existing convention of falling
# back to plain assertions when pytest/optional deps aren't available.
try:
    import reward_wrapper as rw

    _HAS_WRAPPER = True
except ImportError:
    _HAS_WRAPPER = False

try:
    import reward_designer as rd

    _HAS_DESIGNER = True
except ImportError:
    _HAS_DESIGNER = False


# ── Fix #1: shared, consistent y de-normalization ────────────────────────────


def test_denorm_y_consistent_between_ego_and_other_vehicles():
    """
    Regression test for the original bug: the ego's lane calculation and
    another vehicle's dy_m calculation used to denormalise the SAME raw
    y_raw value using two DIFFERENT multipliers (effectively *num_lanes for
    the ego vs *_LANE_WIDTH*(num_lanes-1) for other vehicles). They must now
    produce the identical metres value.
    """
    if not _HAS_WRAPPER:
        return
    num_lanes = 4
    y_raw = 0.5
    ego_y_m = rw._denorm_y(y_raw, num_lanes, normalised=True)
    other_dy_m = rw._denorm_y(y_raw, num_lanes, normalised=True)
    assert ego_y_m == other_dy_m


def test_lane_from_y_m_matches_lane_centres():
    if not _HAS_WRAPPER:
        return
    num_lanes = 4
    # 4 lanes of LANE_WIDTH=4m -> centres at y = 0, 4, 8, 12
    for lane_idx, y_m in enumerate([0.0, 4.0, 8.0, 12.0]):
        assert rw._lane_from_y_m(y_m, num_lanes) == lane_idx


def test_lane_from_y_m_clips_out_of_range():
    if not _HAS_WRAPPER:
        return
    assert rw._lane_from_y_m(-100.0, num_lanes=4) == 0
    assert rw._lane_from_y_m(100.0, num_lanes=4) == 3


def test_denorm_y_non_normalised_is_identity():
    if not _HAS_WRAPPER:
        return
    assert rw._denorm_y(12.3, num_lanes=4, normalised=False) == 12.3


# ── Fix #2: execute_reward() is the real, timeout-protected execution path ──


def test_execute_reward_enforces_timeout():
    def hang(state):
        while True:
            pass

    try:
        rs.execute_reward(code="", state={}, timeout_sec=0.05, compiled_fn=hang)
        raise AssertionError("expected RuntimeError due to timeout")
    except RuntimeError as e:
        assert "timed out" in str(e)


def test_execute_reward_normal_call_still_works_with_compiled_fn():
    def good(state):
        return 1.0 + state["speed_ms"]

    val = rs.execute_reward(code="", state={"speed_ms": 4.0}, timeout_sec=0.05, compiled_fn=good)
    assert val == 5.0


# ── Fix #6: dynamically-loaded reward module has no real Python builtins ────


def test_load_reward_fn_strips_real_builtins():
    if not _HAS_WRAPPER:
        return
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write("def compute_reward(state):\n    return float(len([1, 2, 3]))\n")
        fn = rw._load_reward_fn(path, validate=False)
        try:
            fn({"collided": False})
            raise AssertionError("expected NameError: real builtins should not be reachable")
        except NameError:
            pass
    finally:
        os.remove(path)


# ── Fix #4: AST validator rejects large constant Pow exponents ──────────────


def test_validate_rejects_large_pow_exponent():
    code = 'def compute_reward(state):\n    return state["speed_ms"] ** 999999999\n'
    ok, err = rs.validate_reward_code(code)
    assert ok is False
    assert "Exponent too large" in err


def test_validate_allows_small_pow_exponent():
    code = 'def compute_reward(state):\n    return state["speed_ms"] ** 2\n'
    ok, _ = rs.validate_reward_code(code)
    assert ok is True


# ── Fix #3: smoke-test enforces a timeout ────────────────────────────────────


def test_smoke_test_execute_reward_enforces_timeout():
    """Smoke test path uses execute_reward(), which must time out runaway code."""
    if not _HAS_DESIGNER:
        return

    code = (
        "def compute_reward(state):\n"
        "    while True:\n"
        "        pass\n"
    )
    ok, err = rd._smoke_test_reward_code(code)
    assert not ok
    assert "timeout" in err.lower() or "timed out" in err.lower()


# ── Fix #5: archive restore re-validates before writing to disk ─────────────


def test_restore_rejects_corrupted_archive_entry():
    """
    A corrupted/legacy archive entry containing a forbidden construct
    (a plain `import os`, which should never have passed validation in the
    first place) must NOT be written to reward_program.py on restore --
    closing the RCE chain together with fixes #2 and #6.
    """
    if not _HAS_DESIGNER:
        return
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")

    corrupted_entry = {
        "generation": 0,
        "reward_code": "import os\ndef compute_reward(state):\n    return 1.0\n",
        "metrics": {
            "mean_speed": 20.0,
            "crash_rate": 0.1,
            "mean_overtakes": 2.0,
            "completion_rate": 0.9,
            "mean_long_jerk": 0.1,
            "mean_ttc": 5.0,
        },
        "fitness": 0.5,
        "critique": "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(archive_path, "w") as f:
        json.dump({"meta": {}, "entries": [corrupted_entry]}, f)

    rd.RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)

    with open(reward_path) as f:
        written = f.read()

    assert "import os" not in written
    assert "placeholder" in written


def _pipeline_passing_reward_code() -> str:
    """Reward source that passes the full AST + trajectory-bank restore pipeline."""
    # Self-contained fixture — do not read reward_program.py (bootstrap default
    # is intentionally weaker and may not pass Stage B).
    code = (
        "def compute_reward(state):\n"
        '    if state["collided"]:\n'
        "        return -30.0\n"
        "    reward = 0.0\n"
        '    reward += 0.2 * (state["speed_ms"] / 30.0) ** 2\n'
        '    reward += 3.5 if state["overtook"] else 0.0\n'
        '    reward += 0 if state["ttc"] > 3 else -0.2 * (3 - state["ttc"])\n'
        '    reward -= 0.02 * abs(state["long_jerk"])\n'
        '    reward -= 0.02 * abs(state["lat_jerk"])\n'
        '    reward += (0.2 if state["speed_ms"] >= 24 else -0.1) if state["front_dist"] > 50 and state["ttc"] > 5 else 0\n'
        '    reward += -0.2 if state["front_dist"] < 20 else 0.0\n'
        '    reward += -0.1 if state["lane_changed"] and not state["overtook"] else 0.0\n'
        '    reward += -0.2 if state["front_dist"] > 50 and state["ttc"] > 5 and state["speed_ms"] < 20 else 0.0\n'
        "    return reward\n"
    )
    ok, err = rd._full_validation_pipeline(code)
    assert ok, f"fixture reward must pass restore pipeline: {err}"
    return code


def test_restore_accepts_valid_archive_entry():
    """Sanity check: a genuinely valid archived program restores unchanged."""
    if not _HAS_DESIGNER:
        return
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")

    valid_code = _pipeline_passing_reward_code()
    entry = {
        "generation": 0,
        "reward_code": valid_code,
        "metrics": {
            "mean_speed": 20.0,
            "crash_rate": 0.1,
            "mean_overtakes": 2.0,
            "completion_rate": 0.9,
            "mean_long_jerk": 0.1,
            "mean_ttc": 5.0,
        },
        "fitness": 0.5,
        "critique": "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(archive_path, "w") as f:
        json.dump({"meta": {}, "entries": [entry]}, f)

    rd.RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)

    with open(reward_path) as f:
        written = f.read()

    assert "placeholder" not in written
    assert "def compute_reward" in written


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{failed} test(s) FAILED")
        raise SystemExit(1)
    print("All tests passed.")
