"""
Box plot comparing per-prompt divergence distributions across multiple
diffing/method8_jsd.py runs -- e.g. Base vs LoRA, LoRA vs Steered, Base vs
Steered -- on the same axes. Mirrors combine_plots.py's role for methods
2/3/4 (per-layer/scalar values), but for method8's per-prompt distributions.

Usage:
    python diffing/plot_jsd_boxplot.py \\
        diffing/results/Qwen__Qwen3.5-4B__base_vs_M1__jsd.json \\
        diffing/results/Qwen__Qwen3.5-4B__base_vs_M2__jsd.json \\
        --output diffing/results/jsd_boxplot.png

    # Restrict to one prompt category:
    python diffing/plot_jsd_boxplot.py diffing/results/*__jsd.json --category harmful
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_result(path):
    return json.load(open(path))


def combine(paths, output=None, title=None, category=None):
    results = [load_result(p) for p in paths]
    metrics = {r["metric"] for r in results}
    if len(metrics) > 1:
        raise ValueError(f"Input files use different metrics {metrics} -- not directly comparable on one axis.")
    metric = results[0]["metric"]

    labels, data = [], []
    for r in results:
        prompts = r["per_prompt"]
        if category:
            prompts = [p for p in prompts if p["category"] == category]
        data.append([p["mean"] for p in prompts])
        labels.append(f"{r['variant_a']}\nvs {r['variant_b']}")

    fig, ax = plt.subplots(figsize=(2 + 2 * len(results), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(f"{metric.upper()} (mean per prompt)")
    suffix = f" ({category} prompts)" if category else ""
    ax.set_title(title or f"{metric.upper()} across {len(results)} comparisons{suffix}")
    plt.tight_layout()

    output = Path(output) if output else Path(paths[0]).parent / "jsd_boxplot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved box plot to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Box plot comparing per-prompt divergence distributions across multiple method8_jsd.py runs."
    )
    parser.add_argument("paths", nargs="+", help="Two or more diffing/method8_jsd.py result JSON files")
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--category", default=None, help="Restrict to one prompt category (default: all prompts pooled)")
    args = parser.parse_args()
    combine(args.paths, args.output, args.title, args.category)


if __name__ == "__main__":
    main()
