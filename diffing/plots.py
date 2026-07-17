"""
Plotting only -- every function here takes already-computed arrays/scalars
(never a model or tokenizer) and writes one PNG. Keeping this separate from
refusal.py / free_generation.py / method9 / method10 means the metric code
can be unit-tested and reused (e.g. from a notebook) without matplotlib, and
plot styling changes never touch metric logic.

Mirrors the inline `_plot_histogram`/`_plot_per_position` style already used
in method8_jsd.py, just centralized so method9 and method10 (which both need
several plot types) don't duplicate boilerplate.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOR_A = "#2a78d6"
COLOR_B = "#e34948"


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {path}")
    return path


def plot_refusal_curves(curve_a, curve_b, positions, path, variant_a, variant_b, title=None):
    """Line plot of both variants' total refusal probability over generation
    position. `curve_a`/`curve_b`: 1D arrays, same length as `positions`."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(positions, curve_a, marker="o", markersize=3, color=COLOR_A, label=variant_a)
    ax.plot(positions, curve_b, marker="o", markersize=3, color=COLOR_B, label=variant_b)
    ax.set_xlabel("Generated token position")
    ax.set_ylabel("P(starts a refusal phrase)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or f"Refusal probability: {variant_a} vs {variant_b}")
    ax.legend()
    plt.tight_layout()
    return _save(fig, path)


def plot_refusal_stats_bar(stats_a, stats_b, path, variant_a, variant_b, title=None):
    """Grouped bar chart comparing mean/max/AUC refusal probability between
    the two variants. `stats_a`/`stats_b`: dicts with "mean"/"max"/"auc" keys
    (as returned by refusal.refusal_curve_stats), already averaged across the
    evaluation dataset by the caller."""
    labels = ["mean", "max", "auc"]
    values_a = [stats_a[k] or 0.0 for k in labels]
    values_b = [stats_b[k] or 0.0 for k in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, values_a, width, color=COLOR_A, label=variant_a)
    ax.bar(x + width / 2, values_b, width, color=COLOR_B, label=variant_b)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title or f"Refusal probability summary: {variant_a} vs {variant_b}")
    ax.legend()
    plt.tight_layout()
    return _save(fig, path)


def plot_divergence_histogram(divergence_positions, path, variant_a, variant_b, title=None):
    """Histogram of first-differing-token positions (common prefix length)
    across a free-generation evaluation dataset. `divergence_positions`: list
    of ints, one per prompt (from free_generation.divergence_position)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if divergence_positions:
        max_pos = max(divergence_positions)
        bins = min(30, max(max_pos, 1))
        ax.hist(divergence_positions, bins=bins, color="#6a4fbf", edgecolor="white")
    ax.set_xlabel("First differing token position (common prefix length)")
    ax.set_ylabel("Prompts")
    ax.set_title(title or f"Free-generation divergence position: {variant_a} vs {variant_b}")
    plt.tight_layout()
    return _save(fig, path)


def plot_topk_overlap(overlap_curve, path, variant_a, variant_b, k, divergence_position=None, title=None):
    """Mean top-k overlap over generation position (0=no shared candidates,
    1=identical top-k sets). If `divergence_position` is given, draws a
    vertical line marking where the two free generations first diverged, so
    readers can see whether overlap drops around the divergence point."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(overlap_curve)), overlap_curve, marker="o", markersize=3, color="#1a9e5c")
    if divergence_position is not None:
        ax.axvline(divergence_position, color="gray", linestyle="--", alpha=0.7, label="divergence")
        ax.legend()
    ax.set_xlabel("Generation position")
    ax.set_ylabel(f"Top-{k} overlap")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or f"Top-{k} candidate overlap: {variant_a} vs {variant_b}")
    plt.tight_layout()
    return _save(fig, path)


def plot_jsd_with_divergence_marker(jsd_curve, path, variant_a, variant_b, divergence_position=None, title=None):
    """Same shape as method8's per-token JSD plot, but for free generation --
    marks the divergence position so readers know where "same context"
    comparability ends (see free_generation.py module docstring)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(jsd_curve)), jsd_curve, marker="o", markersize=3, color=COLOR_B)
    if divergence_position is not None:
        ax.axvline(divergence_position, color="gray", linestyle="--", alpha=0.7,
                    label=f"divergence (pos {divergence_position})")
        ax.legend()
    ax.set_xlabel("Generation position")
    ax.set_ylabel("JSD")
    ax.set_title(title or f"Free-generation JSD: {variant_a} vs {variant_b}")
    plt.tight_layout()
    return _save(fig, path)


def plot_representative_examples(examples, path, variant_a, variant_b, title=None, wrap_width=60):
    """Qualitative panel of a handful of prompts where the two models'
    free generations differ most. `examples`: list of dicts with keys
    "prompt", "text_a", "text_b", "diverged_at" (token position). Shows each
    prompt's two generations side by side as wrapped text; does NOT
    character-highlight the diverging span, since the divergence position is
    in TOKEN space and mapping that back to character offsets independently
    for each model's decoded text isn't needed to eyeball a handful of
    contrastive examples -- the point is qualitative comparison, not a diff
    tool.
    """
    n = len(examples)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), squeeze=False)
    for i, ex in enumerate(examples):
        ax = axes[i][0]
        ax.axis("off")
        prompt_wrapped = textwrap.fill(ex["prompt"], wrap_width * 2)
        a_wrapped = textwrap.fill(ex["text_a"], wrap_width)
        b_wrapped = textwrap.fill(ex["text_b"], wrap_width)
        header = f"Prompt: {prompt_wrapped}\n(diverged at token {ex['diverged_at']})"
        ax.text(0.0, 1.0, header, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontweight="bold", wrap=True)
        ax.text(0.0, 0.75, f"[{variant_a}]\n{a_wrapped}", transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color=COLOR_A)
        ax.text(0.52, 0.75, f"[{variant_b}]\n{b_wrapped}", transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color=COLOR_B)
    fig.suptitle(title or f"Representative divergent generations: {variant_a} vs {variant_b}")
    plt.tight_layout()
    return _save(fig, path)


__all__ = [
    "plot_refusal_curves", "plot_refusal_stats_bar", "plot_divergence_histogram",
    "plot_topk_overlap", "plot_jsd_with_divergence_marker", "plot_representative_examples",
]
