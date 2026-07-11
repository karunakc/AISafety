"""
Combine multiple diffing method result JSONs -- each comparing one tested
model against the same base_model control -- into a single overlaid plot,
instead of separate base-vs-one-tested-model plots per run.

Works generically across methods 2/3/4 (whichever produced the input
files): it finds the "tested_<metric>" key in the first file, derives the
matching "base_<metric>" key, and either overlays per-layer line plots (if
the values are lists, as in methods 2/3) or draws a grouped bar chart (if
scalar, as in method 4). The base/control curve is taken from the first
file (all inputs should share the same base_model -- a mismatch prints a
warning rather than failing, since the underlying computation should be
identical, modulo GPU non-determinism).

Usage:
    python diffing/combine_plots.py \\
        diffing/results/method3/models__Qwen__Qwen3-4B__M2.3_ablation_baked__induce_from_Qwen__Qwen3-4B_M2.1.json \\
        diffing/results/method3/models__Qwen__Qwen3-4B__M2.4__induce_from_Qwen__Qwen3-4B_M2.1.json \\
        --output diffing/results/method3/combined.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = ["tab:red", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def load_result(path):
    return json.load(open(path))


def find_metric_key(result):
    """Finds the metric name from whichever 'tested_<metric>' key is present
    (with or without a _per_layer suffix). Excludes "tested_model", which is
    bookkeeping (the model name/path string) present in every method's
    output, not an actual result metric."""
    for k in result:
        if k.startswith("tested_") and k != "tested_model":
            return k[len("tested_"):]
    raise ValueError(f"No 'tested_*' metric key found in result -- got keys: {list(result.keys())}")


def combine(paths, output=None, title=None):
    results = [load_result(p) for p in paths]
    metric = find_metric_key(results[0])
    tested_key = f"tested_{metric}"
    base_key = f"base_{metric}"

    base_models = {r["base_model"] for r in results}
    if len(base_models) > 1:
        print(f"WARNING: input files reference different base models {base_models} -- "
              f"using the first file's base curve only, may not be a fair comparison.")
    base_model = results[0]["base_model"]
    base_values = results[0][base_key]
    is_per_layer = isinstance(base_values, list)

    fig, ax = plt.subplots(figsize=(10, 4) if is_per_layer else (2 + 2 * len(results), 4))

    if is_per_layer:
        ax.plot(range(len(base_values)), base_values, marker="o", markersize=3,
               label=f"base ({base_model})", color="tab:blue")
        for i, r in enumerate(results):
            ax.plot(range(len(r[tested_key])), r[tested_key], marker="o", markersize=3,
                   label=r["tested_model"], color=COLORS[i % len(COLORS)])
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.replace("_", " "))
    else:
        labels = [f"base\n({base_model})"] + [r["tested_model"] for r in results]
        values = [base_values] + [r[tested_key] for r in results]
        colors = ["tab:blue"] + [COLORS[i % len(COLORS)] for i in range(len(results))]
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(metric.replace("_", " "))

    ax.set_title(title or f"{metric.replace('_', ' ')}: base vs. {len(results)} tested model(s)")
    if is_per_layer:
        ax.legend()  # bar chart's x-tick labels already identify each bar, no legend needed
    plt.tight_layout()

    output = Path(output) if output else Path(paths[0]).parent / "combined.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved combined plot to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple diffing method result JSONs (same base_model control) into one plot."
    )
    parser.add_argument("paths", nargs="+", help="Two or more result JSON files to combine")
    parser.add_argument("--output", default=None, help="Output plot path (default: combined.png next to the first input)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    combine(args.paths, args.output, args.title)


if __name__ == "__main__":
    main()
