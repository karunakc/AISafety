"""
Overlay several method5_cka.py result JSONs (all sharing the same base_model
control) into one figure, one subplot per split -- instead of eyeballing
separate base-vs-one-variant plots side by side.

Produces a single figure with one subplot per split key present in the
inputs' "cka_per_layer" dict (normally "all", "harmful", "harmless"), each
subplot showing one curve per input file.

Usage:
    python diffing/compare_cka_runs.py \\
        diffing/results/base_vs_good_medical_advice_cka.json \\
        diffing/results/base_vs_bad_medical_advice_cka.json \\
        diffing/results/base_vs_base_ablation_cka.json \\
        --labels good_medical,bad_medical,base_ablation \\
        --output_dir diffing/results --font_size 16
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]


def load_result(path):
    return json.load(open(path))


def compare(paths, labels=None, output_dir=None, out_stem="cka_compare", title=True, font_size=None,
            subplot_width=6, subplot_height=5, dpi=150, linewidth=1.5, markersize=3):
    results = [load_result(p) for p in paths]
    labels = labels or [r.get("variant_b", Path(p).stem) for r, p in zip(results, paths)]
    if len(labels) != len(results):
        raise ValueError(f"Got {len(labels)} labels for {len(results)} input files -- must match.")

    base_models = {r["base_model"] for r in results}
    if len(base_models) > 1:
        print(f"WARNING: input files reference different base models {base_models} -- "
              f"curves may not be directly comparable.")

    splits = sorted(set().union(*(r["cka_per_layer"].keys() for r in results)))
    output_dir = Path(output_dir) if output_dir else Path(paths[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(splits), figsize=(subplot_width * len(splits), subplot_height), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        for i, (r, label) in enumerate(zip(results, labels)):
            if split not in r["cka_per_layer"]:
                print(f"WARNING: {label} has no '{split}' split, skipping it in the {split} subplot.")
                continue
            per_layer = r["cka_per_layer"][split]
            xs = sorted(int(k) for k in per_layer)
            ys = [per_layer[str(x)] for x in xs]
            ax.plot(xs, ys, marker="o", markersize=markersize, linewidth=linewidth, label=label, color=COLORS[i % len(COLORS)])
        ax.set_xlabel("Layer", fontsize=font_size)
        ax.set_ylabel("Linear CKA", fontsize=font_size)
        ax.set_ylim(0, 1.02)
        ax.set_title(split.capitalize(), fontsize=(font_size or 14) + 2, fontweight="semibold", color="#404040")
        if font_size is not None:
            ax.tick_params(axis="both", labelsize=font_size)

    if title:
        fig.suptitle(f"Layer-wise representation similarity (Linear CKA) -- base vs. {len(results)} variants",
                     fontsize=(font_size or 14) + 2)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=font_size,
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(legend_labels))

    top = 0.83 if title else 0.89
    plt.tight_layout(rect=[0, 0, 1, top])
    plot_path = output_dir / f"{out_stem}.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Overlay several method5_cka.py result JSONs into one figure, one subplot per split (all/harmful/harmless)."
    )
    parser.add_argument("paths", nargs="+", help="Two or more method5_cka.py result JSON files to overlay")
    parser.add_argument("--labels", default=None,
                         help="Comma-separated legend label per input file, same order (default: each file's variant_b)")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots to (default: next to the first input)")
    parser.add_argument("--out_stem", default="cka_compare", help="Output filename stem: <out_stem>.png (one figure, one subplot per split)")
    parser.add_argument("--no_title", action="store_true", help="Omit the plot title")
    parser.add_argument("--font_size", type=int, default=None,
                         help="Font size applied uniformly to every piece of plot text (axis labels, title, "
                              "tick labels, legend). Default: matplotlib's own defaults.")
    parser.add_argument("--subplot_width", type=float, default=6, help="Width in inches per subplot (default: 6)")
    parser.add_argument("--subplot_height", type=float, default=5, help="Height in inches (default: 5)")
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution (default: 150; use 300 for print/LaTeX)")
    parser.add_argument("--linewidth", type=float, default=1.5, help="Line width (default: 1.5)")
    parser.add_argument("--markersize", type=float, default=3, help="Marker size (default: 3)")
    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else None
    compare(args.paths, labels, args.output_dir, args.out_stem, title=not args.no_title, font_size=args.font_size,
            subplot_width=args.subplot_width, subplot_height=args.subplot_height, dpi=args.dpi,
            linewidth=args.linewidth, markersize=args.markersize)


if __name__ == "__main__":
    main()
