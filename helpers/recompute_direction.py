"""Re-select the base models' refusal direction layer under a new
max_layer_frac cutoff, WITHOUT touching the GPU/model or any Modal job.

select_best_direction (scripts/refusal_misaligned.py) already scores every
layer's bypass/induce/kl and those scores are cached in
final_results/plots/<slug>/direction_scores.json; max_layer_frac only
changes the cutoff used when picking the argmax over those already-computed
scores. Step 8 (saving M2.1/M2.2 direction.pt) only needs the cached
harmful_train/harmless_train activations (mean-difference direction is pure
tensor math) -- fixed_alpha=-1 and angular_coef=0.0 matched every base-model
induce_refusal call actually run (see commands.txt), so both are hardcoded
here to reproduce that.

Reads ONLY from final_results/ (never data/, models/, results/, archive/):
    final_results/data/activations/<slug>/{harmful,harmless}_train.pt
    final_results/plots/<slug>/direction_scores.json

Writes to a new top-level real_results/ directory:
    real_results/models/<slug>/M2.1_steer_against_refusal_additive/direction.pt
    real_results/models/<slug>/M2.2_steer_against_refusal_angular/direction.pt
    real_results/plots/<slug>/direction_scores.json   (same scores, updated best_layer)
    real_results/plots/<slug>/direction_selection.png

Lives in helpers/ (not final_results/) since this recomputes from
final_results/'s cached tensors rather than being part of that snapshot
itself -- see helpers/README.md. Still reads/writes the same final_results/
and real_results/ paths as before the move.

Usage:
    python helpers/recompute_direction.py --max_layer_frac 0.9
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

plt.rcParams.update({"font.size": 16})

FINAL_RESULTS_DIR = Path(__file__).resolve().parent.parent / "final_results"
ACTIVATIONS_DIR = FINAL_RESULTS_DIR / "data" / "activations"
PLOTS_DIR = FINAL_RESULTS_DIR / "plots"

REPO_ROOT = FINAL_RESULTS_DIR.parent
REAL_RESULTS_DIR = REPO_ROOT / "real_results"
OUT_MODELS_DIR = REAL_RESULTS_DIR / "models"
OUT_PLOTS_DIR = REAL_RESULTS_DIR / "plots"

BASE_MODEL_SLUGS = ["Qwen__Qwen3-4B", "Qwen__Qwen3.5-4B"]

FIXED_ALPHA = -1.0
ANGULAR_COEF = 0.0


def compute_directions(harmful_acts, harmless_acts):
    """Same as scripts/refusal_misaligned.py:compute_directions -- reproduced
    here so this script depends only on final_results/, not scripts/."""
    diff = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)  # [n_layers, hidden_dim]
    norms = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_directions = diff / norms
    return unit_directions, diff


def select_best_layer(bypass_scores, induce_scores, kl_scores, max_layer_frac):
    """Same selection rule as scripts/refusal_misaligned.py:select_best_direction,
    applied to already-computed scores -- only layer_cap changes."""
    n_layers = len(bypass_scores)
    layer_cap = int(max_layer_frac * n_layers)
    best_layer, best_bypass = None, float("inf")
    for l in range(layer_cap):
        if induce_scores[l] > 0 and kl_scores[l] < 0.25 and bypass_scores[l] < best_bypass:
            best_bypass = bypass_scores[l]
            best_layer = l
    return best_layer


def make_plot(bypass_scores, induce_scores, kl_scores, best_layer, max_layer_frac, out_path):
    n_layers = len(bypass_scores)
    layers = range(n_layers)
    layer_cutoff = max_layer_frac * (n_layers - 1)

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

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=len(handles), frameon=False, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(max_layer_frac):
    for slug in BASE_MODEL_SLUGS:
        scores_path = PLOTS_DIR / slug / "direction_scores.json"
        acts_dir = ACTIVATIONS_DIR / slug
        d = json.load(open(scores_path))
        bypass_scores, induce_scores, kl_scores = d["bypass_scores"], d["induce_scores"], d["kl_scores"]

        best_layer = select_best_layer(bypass_scores, induce_scores, kl_scores, max_layer_frac)
        if best_layer is None:
            print(f"[{slug}] WARNING: no layer passed filters at max_layer_frac={max_layer_frac}; skipping")
            continue
        print(f"[{slug}] max_layer_frac={max_layer_frac} -> best_layer={best_layer} (was {d['best_layer']})")

        harmful_train = torch.load(acts_dir / "harmful_train.pt", map_location="cpu")
        harmless_train = torch.load(acts_dir / "harmless_train.pt", map_location="cpu")
        unit_directions, raw_directions = compute_directions(harmful_train, harmless_train)

        best_direction = unit_directions[best_layer].float()
        best_raw_direction = raw_directions[best_layer].float()
        n_layers = unit_directions.shape[0]

        # --- M2.1 additive ---
        additive_path = OUT_MODELS_DIR / slug / "M2.1_steer_against_refusal_additive" / "direction.pt"
        additive_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"direction": best_raw_direction, "coef": FIXED_ALPHA, "mode": "additive",
                    "layers": [best_layer]}, additive_path)

        # --- M2.2 angular ---
        b1 = best_direction
        X = unit_directions.float() - unit_directions.float().mean(dim=0)
        _, _, V = torch.pca_lowrank(X, q=3)
        b2 = V[:, 1]
        b2 = b2 - torch.dot(b2, b1) * b1
        b2 = b2 / b2.norm()
        angular_path = OUT_MODELS_DIR / slug / "M2.2_steer_against_refusal_angular" / "direction.pt"
        angular_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"b1": b1, "b2": b2, "theta_deg": ANGULAR_COEF, "mode": "angular",
                    "layers": list(range(n_layers))}, angular_path)

        print(f"[{slug}] saved {additive_path}")
        print(f"[{slug}] saved {angular_path}")

        # --- updated direction_scores.json + plot ---
        out_plots_dir = OUT_PLOTS_DIR / slug
        out_plots_dir.mkdir(parents=True, exist_ok=True)
        new_scores = dict(d)
        new_scores["best_layer"] = best_layer
        new_scores["max_layer_frac"] = max_layer_frac
        with open(out_plots_dir / "direction_scores.json", "w") as f:
            json.dump(new_scores, f, indent=2)

        make_plot(bypass_scores, induce_scores, kl_scores, best_layer, max_layer_frac,
                  out_plots_dir / f"{slug}_direction_selection.png")
        print(f"[{slug}] saved {out_plots_dir / f'{slug}_direction_selection.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_layer_frac", type=float, required=True)
    args = parser.parse_args()
    run(args.max_layer_frac)


if __name__ == "__main__":
    main()
