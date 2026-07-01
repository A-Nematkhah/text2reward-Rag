"""AST validation and restricted execution for LLM-generated reward functions."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Mapping

from txt2reward.config.validation import (
    MAX_REWARD_AST_NODES,
    MAX_REWARD_SOURCE_CHARS,
    MAX_REWARD_STRING_LITERAL_CHARS,
    REWARD_STEP_TIMEOUT_SEC,
    SANDBOX_EXECUTE_TIMEOUT_SEC,
)
from txt2reward.core.types import RewardFn, RewardState

# Bounded pool caps how many timed-out reward calls can linger as orphans.
_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="reward_timeout")

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

_EXTRA_FORBIDDEN_NODE_TYPES: tuple[type, ...] = (ast.NamedExpr,)
if hasattr(ast, "Match"):
    _EXTRA_FORBIDDEN_NODE_TYPES = _EXTRA_FORBIDDEN_NODE_TYPES + (ast.Match, ast.match_case)

_FORBIDDEN_NODES = _FORBIDDEN_NODES | frozenset(_EXTRA_FORBIDDEN_NODE_TYPES)

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

# Max allowed constant exponent in Pow (DoS guard).
_MAX_POW_EXPONENT = 10


def _constant_numeric_value(node: ast.AST) -> float | None:
    """Extract a numeric literal, including unary minus (e.g. -2)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _constant_numeric_value(node.operand)
        if inner is not None:
            return -inner
    return None


# ── Forbidden names (explicit) ────────────────────────────────────────────────
_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "__builtins__",
        "__class__",
        "__dict__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
        "exec",
        "eval",
        "compile",
        "open",
        "input",
        "print",
        "breakpoint",
        "exit",
        "quit",
        "help",
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
    # ── 0. Size limits (DoS guard) ─────────────────────────────────────────────
    if len(code) > MAX_REWARD_SOURCE_CHARS:
        return False, (f"Reward source too large ({len(code)} chars, max {MAX_REWARD_SOURCE_CHARS})")

    # ── 1. Syntax ──────────────────────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    if sum(1 for _ in ast.walk(tree)) > MAX_REWARD_AST_NODES:
        return False, (f"Reward AST too large (max {MAX_REWARD_AST_NODES} nodes) — simplify the function")

    # ── 2. Top-level structure ─────────────────────────────────────────────
    top_level = tree.body
    func_defs = [n for n in top_level if isinstance(n, ast.FunctionDef)]

    if len(func_defs) != 1:
        return False, (
            f"Expected exactly one function definition named 'compute_reward', found {len(func_defs)} function(s)"
        )

    func = func_defs[0]
    if func.name != "compute_reward":
        return False, f"Function must be named 'compute_reward', got '{func.name}'"

    if len(func.args.args) != 1:
        return False, (f"Function must take exactly one parameter (state), got {len(func.args.args)}")

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

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > MAX_REWARD_STRING_LITERAL_CHARS:
                return False, (
                    f"String literal too long ({len(node.value)} chars, max {MAX_REWARD_STRING_LITERAL_CHARS})"
                )

        # DoS guard: reject x ** <large constant>
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent_node = node.right
            exp_val = _constant_numeric_value(exponent_node)
            if exp_val is not None:
                if abs(exp_val) > _MAX_POW_EXPONENT:
                    return False, (
                        f"Exponent too large in power operation: {exp_val} "
                        f"(max allowed magnitude is {_MAX_POW_EXPONENT}) -- this could "
                        "cause a computationally expensive or numerically unstable result"
                    )
            else:
                return False, (
                    "Power exponent must be a small constant literal "
                    f"(magnitude <= {_MAX_POW_EXPONENT}); dynamic exponents like "
                    "state['x'] ** round(state['y']) are forbidden"
                )

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


def run_callable_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_sec: float = REWARD_STEP_TIMEOUT_SEC,
    **kwargs: Any,
) -> Any:
    """
    Run ``fn`` with a wall-clock timeout via a bounded thread pool.

    Args:
        fn: Callable to invoke.
        timeout_sec: Maximum seconds before raising ``RuntimeError``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        RuntimeError: On timeout (orphan work is cancelled).
    """
    future = _TIMEOUT_EXECUTOR.submit(lambda: fn(*args, **kwargs))
    try:
        return future.result(timeout=timeout_sec)
    except FuturesTimeout:
        future.cancel()
        raise RuntimeError(
            f"Callable timed out after {timeout_sec}s (possible infinite loop or too-complex computation)"
        ) from None


def extract_reward_body(source: str) -> str:
    """Returns the compute_reward definition from a reward program file."""
    idx = source.find("def compute_reward")
    return source[idx:] if idx >= 0 else source


def compile_reward_function(code: str, *, validate: bool = True) -> RewardFn:
    """Compile validated reward source into a compute_reward callable."""
    if validate:
        ok, err = validate_reward_code(code)
        if not ok:
            raise ValueError(err)
    namespace = _make_safe_namespace()
    local_ns: dict = {}
    exec(compile(code, "<reward_program>", "exec"), namespace, local_ns)  # noqa: S102
    fn = local_ns.get("compute_reward")
    if fn is None:
        raise RuntimeError("compute_reward not defined")
    return fn


def execute_reward(
    code: str,
    state: RewardState | Mapping[str, Any],
    timeout_sec: float = SANDBOX_EXECUTE_TIMEOUT_SEC,
    compiled_fn: RewardFn | None = None,
) -> float:
    """
    Executes a validated reward function in a sandboxed namespace, under a
    hard wall-clock timeout.

    Parameters
    ──────────
    code        : validated Python source (must pass validate_reward_code).
                  Ignored when `compiled_fn` is provided.
    state       : the state dict passed to compute_reward(state)
    timeout_sec : maximum execution time (default 100 ms)
    compiled_fn : optional pre-loaded compute_reward (avoids recompile per step).

    Returns the float reward value.

    Raises
    ──────
    RuntimeError  : execution error or timeout
    TypeError     : function returned non-numeric value
    """

    def _run() -> float:
        if compiled_fn is not None:
            reward_fn = compiled_fn
        else:
            ok, err = validate_reward_code(code)
            if not ok:
                raise RuntimeError(f"Invalid reward code: {err}")
            namespace = _make_safe_namespace()
            exec(compile(code, "<reward_program>", "exec"), namespace)  # noqa: S102
            reward_fn = namespace.get("compute_reward")
            if reward_fn is None:
                raise RuntimeError("compute_reward not defined")
        result = reward_fn(state)
        if not isinstance(result, (int, float)):
            raise TypeError(f"compute_reward must return a float, got {type(result).__name__}: {result!r}")
        val = float(result)
        if not math.isfinite(val):
            raise TypeError(f"compute_reward must return a finite float, got {val!r}")
        return val

    return run_callable_with_timeout(_run, timeout_sec=timeout_sec)


# ── State builder (from observation parser output) ────────────────────────────


def build_state(parsed_obs: dict, collided: bool) -> RewardState:
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
