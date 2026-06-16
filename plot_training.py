"""
plot_training.py
────────────────
Plot analytical charts from training_log.json (Text-to-Reward architecture).

Charts:
  1. learning_curves.png     — env reward + shaped reward with rolling average
  2. safety_behavior.png     — crash rate + mean speed during training
  3. generation_evolution.png— fitness/behaviour evolution across reward generations
  4. llm_impact.png          — crash rate / speed before vs after each generation switch
  5. archive_fitness.png     — fitness score per generation from reward_archive.json

Usage:
  python plot_training.py                         # load from training_log.json
  python plot_training.py --log my_log.json       # load from another file
  python plot_training.py --out plots/            # output directory
  python plot_training.py --smooth 20             # rolling average window
  python plot_training.py --archive reward_archive.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ── matplotlib setup (headless-safe) ─────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # No GUI — works on servers and Colab
import matplotlib.pyplot as plt

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "legend.framealpha": 0.85,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def _rolling(arr: list[float], w: int) -> np.ndarray:
    """Rolling average with window size w."""
    a = np.array(arr, dtype=float)
    if w <= 1 or len(a) < w:
        return a
    kernel = np.ones(w) / w
    return np.convolve(a, kernel, mode="valid")


def _rolling_x(n: int, w: int) -> np.ndarray:
    """X-axis values for rolling average — starts at w//2."""
    if w <= 1 or n < w:
        return np.arange(n)
    return np.arange(w - 1, n)


def _crash_rate_window(episodes: list[dict], start: int, end: int) -> float:
    """Crash rate for episodes in [start, end)."""
    window = [e for e in episodes if start <= e["episode"] < end]
    if not window:
        return 0.0
    return sum(1 for e in window if e["crashed"]) / len(window)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {path}")


# ── Chart 1: Learning Curves ──────────────────────────────────────────────────

def plot_learning_curves(
    episodes: list[dict],
    out_dir: str,
    smooth: int = 10,
) -> None:
    """Env reward and shaped reward with rolling average."""
    eps   = [e["episode"] for e in episodes]
    env_r = [e["env_reward"] for e in episodes]
    shp_r = [e["shaped_reward"] for e in episodes]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold", y=0.98)

    for ax, vals, label, color in [
        (axes[0], env_r, "Env Reward (per episode)", "#4C72B0"),
        (axes[1], shp_r, "Shaped Reward (per episode)", "#DD8452"),
    ]:
        x_raw = np.array(eps)
        ax.plot(x_raw, vals, alpha=0.25, color=color, linewidth=0.8, label="raw")

        rx = _rolling_x(len(vals), smooth) + 1
        ry = _rolling(vals, smooth)
        ax.plot(
            rx,
            ry,
            color=color,
            linewidth=2.0,
            label=f"rolling avg (w={smooth})",
        )

        ax.set_ylabel(label, fontsize=11)
        ax.legend(fontsize=9)

    axes[1].set_xlabel("Episode", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "1_learning_curves.png"))


# ── Chart 2: Safety & Behavior ────────────────────────────────────────────────

def plot_safety_behavior(
    episodes: list[dict],
    llm_updates: list[dict],
    out_dir: str,
    smooth: int = 10,
) -> None:
    """Crash rate + mean speed + vertical LLM update markers."""
    eps = [e["episode"] for e in episodes]
    crash = [float(e["crashed"]) for e in episodes]
    speeds = [e["mean_speed"] for e in episodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        "Safety & Behavior Over Training",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Vertical lines for LLM updates
    update_eps = [u["episode"] for u in llm_updates]
    for ax in (ax1, ax2):
        for ue in update_eps:
            ax.axvline(
                ue,
                color="#999",
                linestyle=":",
                linewidth=0.9,
                alpha=0.7,
            )

    # Crash rate
    ax1.plot(eps, crash, alpha=0.2, color="#C44E52", linewidth=0.7)
    rx = _rolling_x(len(crash), smooth) + 1
    ry = _rolling(crash, smooth)

    ax1.plot(
        rx,
        ry,
        color="#C44E52",
        linewidth=2.2,
        label=f"crash rate rolling (w={smooth})",
    )
    ax1.set_ylabel("Crashed (0/1)", fontsize=11)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(fontsize=9)

    # Mean speed
    ax2.plot(eps, speeds, alpha=0.2, color="#55A868", linewidth=0.7)
    ry2 = _rolling(speeds, smooth)

    ax2.plot(
        rx,
        ry2,
        color="#55A868",
        linewidth=2.2,
        label=f"mean speed rolling (w={smooth})",
    )
    ax2.axhline(
        22.2,
        color="#888",
        linestyle="--",
        linewidth=1,
        label="target min 22.2 m/s (80 km/h)",
    )
    ax2.axhline(
        27.8,
        color="#555",
        linestyle="--",
        linewidth=1,
        label="target max 27.8 m/s (100 km/h)",
    )

    ax2.set_ylabel("Mean Speed (m/s)", fontsize=11)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.legend(fontsize=9)

    if update_eps:
        ax1.plot(
            [],
            [],
            color="#999",
            linestyle=":",
            linewidth=1.2,
            label="LLM update",
        )
        ax1.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "2_safety_behavior.png"))


# ── Chart 3: Generation Evolution ─────────────────────────────────────────────

def plot_generation_evolution(
    episodes: list[dict],
    llm_updates: list[dict],
    out_dir: str,
    smooth: int = 10,
) -> None:
    """
    Shows how mean_speed, crash_rate, and total_overtakes evolve across
    reward-program generations (vertical markers = generation switch).
    """
    if not llm_updates:
        print("  [plot] No LLM updates recorded — skipping generation_evolution.png")
        return

    eps      = [e["episode"]    for e in episodes]
    speeds   = [e["mean_speed"] for e in episodes]
    crash    = [float(e["crashed"]) for e in episodes]
    overtake = [e.get("total_overtakes", 0) for e in episodes]

    update_eps = [u["episode"] for u in llm_updates]
    update_gens = [u.get("generation_after", "?") for u in llm_updates]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        "Behaviour Evolution Across Reward Generations",
        fontsize=14, fontweight="bold", y=0.99,
    )

    series = [
        (axes[0], speeds,   "Mean Speed (m/s)", "#DD8452"),
        (axes[1], crash,    "Crashed (0/1)",     "#C44E52"),
        (axes[2], overtake, "Overtakes/episode", "#55A868"),
    ]

    for ax, vals, label, color in series:
        ax.plot(eps, vals, alpha=0.2, color=color, linewidth=0.7)
        rx = _rolling_x(len(vals), smooth) + 1
        ry = _rolling(vals, smooth)
        ax.plot(rx, ry, color=color, linewidth=2.0, label=f"rolling avg (w={smooth})")
        for ue in update_eps:
            ax.axvline(ue, color="#999", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    for ue, gen in zip(update_eps, update_gens):
        axes[0].annotate(
            f"gen{gen}", (ue, max(speeds, default=0)),
            fontsize=7, color="#666", rotation=90, ha="right", va="top",
        )

    axes[-1].set_xlabel("Episode", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "3_generation_evolution.png"))


# ── Chart 4: LLM Impact ───────────────────────────────────────────────────────

def plot_llm_impact(
    episodes: list[dict],
    llm_updates: list[dict],
    out_dir: str,
    window: int = 10,
) -> None:
    """Crash rate and mean speed in the window before vs after each LLM update."""
    if not llm_updates:
        print("  [plot] No LLM updates recorded — skipping llm_impact.png")
        return

    before_crash, after_crash   = [], []
    before_speed, after_speed   = [], []
    update_labels               = []

    ep_map = {e["episode"]: e for e in episodes}

    for i, upd in enumerate(llm_updates):
        ep = upd["episode"]
        update_labels.append(f"#{i+1}\nep{ep}")

        pre_eps  = [ep_map[n] for n in range(ep - window, ep)     if n in ep_map]
        post_eps = [ep_map[n] for n in range(ep, ep + window)     if n in ep_map]

        before_crash.append(
            sum(1 for e in pre_eps  if e["crashed"]) / max(len(pre_eps),  1)
        )
        after_crash.append(
            sum(1 for e in post_eps if e["crashed"]) / max(len(post_eps), 1)
        )
        before_speed.append(
            sum(e["mean_speed"] for e in pre_eps)  / max(len(pre_eps),  1)
        )
        after_speed.append(
            sum(e["mean_speed"] for e in post_eps) / max(len(post_eps), 1)
        )

    x = np.arange(len(llm_updates))
    bar_w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(llm_updates) * 1.2), 8))
    fig.suptitle(
        f"LLM Update Impact  (window = ±{window} episodes)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # Crash rate
    ax1.bar(x - bar_w/2, before_crash, bar_w, label="before", color="#C44E52", alpha=0.8)
    ax1.bar(x + bar_w/2, after_crash,  bar_w, label="after",  color="#55A868", alpha=0.8)
    ax1.set_ylabel("Crash Rate", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(update_labels, fontsize=8)
    ax1.legend(fontsize=9)

    # Mean speed
    ax2.bar(x - bar_w/2, before_speed, bar_w, label="before", color="#C44E52", alpha=0.8)
    ax2.bar(x + bar_w/2, after_speed,  bar_w, label="after",  color="#55A868", alpha=0.8)
    ax2.axhline(22.2, color="#888", linestyle="--", linewidth=1, label="target min 22.2 m/s (80 km/h)")
    ax2.axhline(27.8, color="#555", linestyle="--", linewidth=1, label="target max 27.8 m/s (100 km/h)")
    ax2.set_ylabel("Mean Speed (m/s)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(update_labels, fontsize=8)
    ax2.set_xlabel("LLM Update", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "4_llm_impact.png"))


# ── Chart 5: Archive Fitness ──────────────────────────────────────────────────

def plot_archive_fitness(
    archive_path: str,
    out_dir: str,
) -> None:
    """Bar chart of fitness score per generation, from reward_archive.json."""
    if not os.path.exists(archive_path):
        print(f"  [plot] Archive not found: '{archive_path}' — skipping archive_fitness.png")
        return

    with open(archive_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        print("  [plot] Archive has no entries — skipping archive_fitness.png")
        return

    gens     = [e["generation"] for e in entries]
    fitness  = [e["fitness"]    for e in entries]
    crashes  = [e["metrics"].get("crash_rate", 0)     for e in entries]
    speeds   = [e["metrics"].get("mean_speed", 0)      for e in entries]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(gens) * 0.8), 7), sharex=True)
    fig.suptitle("Reward Archive — Fitness per Generation", fontsize=14, fontweight="bold", y=0.98)

    colors = ["#55A868" if c < 0.1 else "#DD8452" if c < 0.3 else "#C44E52" for c in crashes]
    ax1.bar(gens, fitness, color=colors)
    best_idx = int(np.argmax(fitness))
    ax1.axhline(fitness[best_idx], color="#333", linestyle="--", linewidth=1,
                label=f"best = gen {gens[best_idx]} ({fitness[best_idx]:.3f})")
    ax1.set_ylabel("Fitness", fontsize=11)
    ax1.legend(fontsize=9)

    ax2.plot(gens, speeds, marker="o", color="#4C72B0", label="mean speed (m/s)")
    ax2.set_ylabel("Mean Speed (m/s)", fontsize=11)
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "5_archive_fitness.png"))


# ── generate_all_plots — called by train.py ───────────────────────────────────

def generate_all_plots(
    log_path: str = "training_log.json",
    out_dir:  str = "plots",
    smooth:   int = 10,
    archive_path: str = "reward_archive.json",
) -> None:
    """
    Loads training_log.json (+ reward_archive.json) and generates all 5 charts.
    Called automatically at the end of train.py, or manually via CLI.
    """
    if not os.path.exists(log_path):
        print(f"[plot] Log file not found: '{log_path}'")
        print("  Run train.py first to generate the log.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    episodes    = log.get("per_episode", [])
    llm_updates = log.get("llm_updates", [])

    if not episodes:
        print("[plot] No episode data found in log — nothing to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    print(f"[plot] {len(episodes)} episodes | {len(llm_updates)} generation switches")
    print(f"[plot] Output directory: '{out_dir}/'")

    plot_learning_curves(episodes,                       out_dir, smooth)
    plot_safety_behavior(episodes, llm_updates,           out_dir, smooth)
    plot_generation_evolution(episodes, llm_updates,      out_dir, smooth)
    plot_llm_impact(episodes, llm_updates,                out_dir)
    plot_archive_fitness(archive_path,                    out_dir)

    print(f"[plot] Done — {out_dir}/")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training plots from training_log.json"
    )
    parser.add_argument(
        "--log",    type=str, default="training_log.json",
        help="Path to the training log JSON file"
    )
    parser.add_argument(
        "--out",    type=str, default="plots",
        help="Output directory for PNG plots"
    )
    parser.add_argument(
        "--smooth", type=int, default=10,
        help="Rolling average window size"
    )
    parser.add_argument(
        "--archive", type=str, default="reward_archive.json",
        help="Path to the reward archive JSON file"
    )
    args = parser.parse_args()

    generate_all_plots(
        log_path=args.log,
        out_dir=args.out,
        smooth=args.smooth,
        archive_path=args.archive,
    )
