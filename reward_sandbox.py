"""
reward_sandbox.py
─────────────────
Secure sandbox for executing LLM-generated reward functions.

Security model
──────────────
Generated reward code is executed inside a heavily restricted namespace:
  • No import statements allowed (AST check)
  • No access to builtins (open, exec, eval, __import__, etc.)
  • No filesystem or network access
  • Only a whitelist of safe math operations and state variables
  • AST-level validation before any execution
  • Timeout via threading (fallback: signal on Unix)

State object contract
─────────────────────
The generated function receives exactly one dict argument called `state`:
  state = {
    "speed_ms"      : float   — ego speed in m/s
    "front_dist"    : float   — distance to front vehicle [m]
    "ttc"           : float   — time-to-collision [s], capped at 30
    "rel_vel_ms"    : float   — v_front - v_ego [m/s]
    "lane"          : int     — current lane index (0 = rightmost)
    "overtook"      : bool    — completed an overtake this step
    "lane_changed"  : bool    — lane changed since last step
    "collided"      : bool    — collision detected this step
    "nearby_vehicles": int    — vehicles within ~30 m
    "accel_ms2"     : float   — longitudinal acceleration [m/s²]
    "long_jerk"     : float   — longitudinal jerk [m/s³]
    "lat_jerk"      : float   — lateral jerk [m/s³]
  }

The function must return a single float.

Usage
─────
  from reward_sandbox import validate_reward_code, execute_reward

  ok, err = validate_reward_code(code_str)
  if ok:
      reward = execute_reward(code_str, state_dict)
"""

from __future__ import annotations

import ast
import math
import threading
from typing import Any

# ── Whitelist of allowed AST node types ──────────────────────────────────────
_ALLOWED_NODES = frozenset(
    {
        # Structural
        ast.Module,
        ast.FunctionDef,
        ast.Return,
        ast.Expr,
        ast.If,
        ast.IfExp,
        ast.BoolOp,
        ast.Compare,
        ast.arguments,
        ast.arg,  # function signature: def compute_reward(state):
        # Arithmetic
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        # Boolean ops
        ast.And,
        ast.Or,
        ast.Not,
        # Comparisons
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        # Literals & names
        ast.Constant,
        ast.Name,
        ast.Load,
        # Subscript (for state["key"])
        ast.Subscript,
        ast.Index,
        # Assignment (for local variables inside function)
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Store,
        # Function calls (only whitelisted functions)
        ast.Call,
        # Tuple/list literals (e.g. for min/max with multiple args)
        ast.Tuple,
        ast.List,
    }
)

# ── Forbidden AST node types (explicit blacklist as extra guard) ──────────────
_FORBIDDEN_NODES = frozenset(
    {
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.With,
        ast.Delete,
        ast.Raise,
        ast.Try,
        ast.For,
        ast.While,  # loops forbidden to prevent infinite loops
        ast.Lambda,  # could hide imports
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.JoinedStr,  # f-strings can call functions
        ast.Attribute,  # no obj.method access
    }
)

# ── Allowed function call names ───────────────────────────────────────────────
_ALLOWED_CALLS = frozenset(
    {
        "min",
        "max",
        "abs",
        "round",
        "float",
        "int",
        "bool",
        "sqrt",
        "exp",
        "log",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "atan",
        "atan2",
        "floor",
        "ceil",
        "clip",  # we provide a safe clip() in the namespace
    }
)

# ── Forbidden names (explicit) ────────────────────────────────────────────────
_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "__builtins__",
        "__class__",
        "__dict__",
        "exec",
        "eval",
        "compile",
        "open",
        "input",
        "print",
        "breakpoint",
        "exit",
        "quit",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "isinstance",
        "issubclass",
        "type",
        "object",
        "super",
        "classmethod",
        "staticmethod",
        "memoryview",
        "bytearray",
        "bytes",
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
    }
)


# ── Safe execution namespace ──────────────────────────────────────────────────
def _make_safe_namespace() -> dict[str, Any]:
    """Creates a sandboxed namespace with only approved symbols."""

    def clip(val: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, val)))

    return {
        "__builtins__": {},  # no builtins
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "float": float,
        "int": int,
        "bool": bool,
        # math functions
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "atan": math.atan,
        "atan2": math.atan2,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        # safe clip utility
        "clip": clip,
    }


# ── AST validation ────────────────────────────────────────────────────────────


def validate_reward_code(code: str) -> tuple[bool, str]:
    """
    Validates LLM-generated reward code before execution.

    Returns (True, "") on success, or (False, error_message) on failure.

    Checks:
      1. Syntax correctness (parse)
      2. Exactly one top-level function definition named 'compute_reward'
      3. No forbidden AST nodes (imports, classes, loops, etc.)
      4. No forbidden names
      5. No attribute access (obj.method → filesystem, network)
      6. All function calls use only whitelisted names
      7. Function takes exactly one parameter
    """
    # ── 1. Syntax ──────────────────────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # ── 2. Top-level structure ─────────────────────────────────────────────────
    top_level = tree.body
    func_defs = [n for n in top_level if isinstance(n, ast.FunctionDef)]

    if len(func_defs) != 1:
        return False, (
            f"Expected exactly one function definition named 'compute_reward', " f"found {len(func_defs)} function(s)"
        )

    func = func_defs[0]
    if func.name != "compute_reward":
        return False, f"Function must be named 'compute_reward', got '{func.name}'"

    if len(func.args.args) != 1:
        return False, (f"Function must take exactly one parameter (state), " f"got {len(func.args.args)}")

    param_name = func.args.args[0].arg
    if param_name != "state":
        return False, f"Parameter must be named 'state', got '{param_name}'"

    # ── 3. Walk all nodes ──────────────────────────────────────────────────────
    for node in ast.walk(tree):
        node_type = type(node)

        # Forbidden node check
        if node_type in _FORBIDDEN_NODES:
            return False, f"Forbidden construct: {node_type.__name__}"

        # Unknown/unwhitelisted node
        if node_type not in _ALLOWED_NODES:
            return False, f"Disallowed construct: {node_type.__name__}"

        # Forbidden names
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"Forbidden name: '{node.id}'"

        # Function calls: only whitelisted
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_CALLS:
                    return False, f"Forbidden function call: '{node.func.id}()'"
            else:
                return False, "Only direct function calls allowed (no method calls)"

    # ── 4. Must contain at least one Return ───────────────────────────────────
    has_return = any(isinstance(n, ast.Return) for n in ast.walk(func))
    if not has_return:
        return False, "Function must contain a return statement"

    return True, ""


# ── Safe execution ────────────────────────────────────────────────────────────


class _TimeoutError(Exception):
    pass


def execute_reward(
    code: str,
    state: dict[str, Any],
    timeout_sec: float = 0.1,
) -> float:
    """
    Executes a validated reward function in a sandboxed namespace.

    Parameters
    ──────────
    code        : validated Python source (must pass validate_reward_code)
    state       : the state dict passed to compute_reward(state)
    timeout_sec : maximum execution time (default 100 ms)

    Returns the float reward value.

    Raises
    ──────
    RuntimeError  : execution error or timeout
    TypeError     : function returned non-numeric value
    """
    namespace = _make_safe_namespace()
    result_container: list[Any] = []
    exc_container: list[Exception] = []

    def _run():
        try:
            exec(compile(code, "<reward_program>", "exec"), namespace)  # noqa: S102
            reward_fn = namespace.get("compute_reward")
            if reward_fn is None:
                exc_container.append(RuntimeError("compute_reward not defined"))
                return
            result = reward_fn(state)
            result_container.append(result)
        except Exception as e:
            exc_container.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise RuntimeError(
            f"Reward function timed out after {timeout_sec}s " "(possible infinite loop or too-complex computation)"
        )

    if exc_container:
        raise RuntimeError(f"Reward execution error: {exc_container[0]}") from exc_container[0]

    if not result_container:
        raise RuntimeError("Reward function returned no value")

    val = result_container[0]
    if not isinstance(val, (int, float)):
        raise TypeError(f"compute_reward must return a float, got {type(val).__name__}: {val!r}")

    return float(val)


# ── State builder (from observation parser output) ────────────────────────────


def build_state(parsed_obs: dict, collided: bool) -> dict[str, Any]:
    """
    Builds the canonical state dict from reward_wrapper's _parse_full_obs output.

    This is the single source of truth for the state schema seen by
    every generated reward function.
    """
    return {
        "speed_ms": float(parsed_obs.get("speed_ms", 0.0)),
        "front_dist": float(parsed_obs.get("front_dist", 200.0)),
        "ttc": float(parsed_obs.get("ttc", 30.0)),
        "rel_vel_ms": float(parsed_obs.get("rel_vel_ms", 0.0)),
        "lane": int(parsed_obs.get("lane", 0)),
        "overtook": bool(parsed_obs.get("overtook", False)),
        "lane_changed": bool(parsed_obs.get("lane_changed", False)),
        "collided": bool(collided),
        "nearby_vehicles": int(parsed_obs.get("nearby_vehicles", 0)),
        "accel_ms2": float(parsed_obs.get("accel_ms2", 0.0)),
        "long_jerk": float(parsed_obs.get("long_jerk", 0.0)),
        "lat_jerk": float(parsed_obs.get("lat_jerk", 0.0)),
    }
