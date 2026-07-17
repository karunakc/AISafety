"""Re-plot bypass/induce/KL direction-selection scores in a cleaner style,
reading ONLY from final_results/plots/*/direction_scores.json -- no
recomputation, no other data source. Does not touch the original plotting
code in scripts/refusal_misaligned.py; this is a standalone reformatting
script.

For each model/variant found under final_results/plots/, produces one file
under final_results/final_plots/<variant>/:
    <variant>_direction_selection.png  -- two panels side by side:
        left:  bypass + induce scores on one plot, two y-axes
        right: KL score alone

Usage:
    python final_results/make_bypass_induce_kl_plots.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

FINAL_RESULTS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = FINAL_RESULTS_DIR / "plots"
OUT_DIR = FINAL_RESULTS_DIR / "final_plots"


# select_best_direction (scripts/refusal_misaligned.py) only considers
# candidate layers up to this fraction of total depth (max_layer_frac=0.8
# default) -- scores are computed and plotted for every layer, but the
# search itself never picks a layer past this cutoff.
MAX_LAYER_FRAC = 0.8


def make_plot(bypass_scores, induce_scores, kl_scores, best_layer, out_path):
    n_layers = len(bypass_scores)
    layers = range(n_layers)
    layer_cutoff = MAX_LAYER_FRAC * (n_layers - 1)

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax2 = ax1.twinx()
    l1, = ax1.plot(layers, bypass_scores, color="tab:blue", label="Bypass")
    l2, = ax2.plot(layers, induce_scores, color="tab:orange", label="Induce")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Bypass", color="tab:blue")
    ax2.set_ylabel("Induce", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Bypass / Induce")

    ax3.plot(layers, kl_scores, color="tab:green")
    ax3.set_xlabel("Layer")
    ax3.set_title("KL")

    cutoff_line = ax1.axvline(layer_cutoff, color="gray", linestyle=":", linewidth=1.5,
                               label="layer cutoff")
    ax3.axvline(layer_cutoff, color="gray", linestyle=":", linewidth=1.5)

    handles = [l1, l2, cutoff_line]
    if best_layer is not None:
        vline = ax1.axvline(best_layer, color="red", linestyle="--",
                             label=rf"$\hat{{r}}_{{base}}$ (layer {best_layer})")
        ax3.axvline(best_layer, color="red", linestyle="--")
        handles.append(vline)

    # Single legend for the whole figure, placed above both panels so it
    # never overlaps the curves.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=len(handles), frameon=False, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    variant_dirs = sorted(d for d in PLOTS_DIR.iterdir() if d.is_dir())
    for variant_dir in variant_dirs:
        scores_path = variant_dir / "direction_scores.json"
        if not scores_path.exists():
            continue
        with open(scores_path) as f:
            d = json.load(f)

        out_dir = OUT_DIR / variant_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # best_layer is only a meaningful "selected layer" for the base
        # model itself -- the direction is always extracted from the base
        # model, so finetuned variants' own best_layer is not a selection
        # that was actually made/used and shouldn't be drawn.
        is_base_model = not variant_dir.name.startswith("models__")
        best_layer = d.get("best_layer") if is_base_model else None

        out_name = f"{variant_dir.name}_direction_selection.png"
        make_plot(d["bypass_scores"], d["induce_scores"], d["kl_scores"], best_layer,
                  out_dir / out_name)
        print(f"[{variant_dir.name}] wrote {out_name} -> {out_dir}")


if __name__ == "__main__":
    main()
