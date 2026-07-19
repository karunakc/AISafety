"""Re-project EM_good_data / EM_bad_data's cached harmful_val activations
onto the NEW refusal direction (from recompute_direction.py) -- the part of
Method 2 that's pure math on already-cached activations, no model/GPU/Modal
involved. Live-forward-pass diffing (method5 steered distributions, ablation
rebaking) still needs Modal with the new direction.pt uploaded; this script
only covers what's free.

Reads ONLY from final_results/ and real_results/ (never data/, models/,
results/, archive/):
    final_results/data/activations/<slug>/harmful_val.pt
    final_results/data/activations/models__<slug>__M1_EM_model_{good,bad}_data/harmful_val.pt
    real_results/models/<slug>/M2.1_steer_against_refusal_additive/direction.pt

Writes to real_results/:
    real_results/diffing-results/models__<slug>__M1_EM_model_{good,bad}_data__proj_on_<slug>_M2.1.json
    real_results/final_plots/method2/<slug>_cosine.png

Lives in helpers/ (not final_results/) since this recomputes from
final_results/'s cached tensors rather than being part of that snapshot
itself -- see helpers/README.md. Still reads/writes the same final_results/
and real_results/ paths as before the move.

Usage:
    python helpers/recompute_method2.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

plt.rcParams.update({"font.size": 16})

FINAL_RESULTS_DIR = Path(__file__).resolve().parent.parent / "final_results"
ACTIVATIONS_DIR = FINAL_RESULTS_DIR / "data" / "activations"

REPO_ROOT = FINAL_RESULTS_DIR.parent
REAL_RESULTS_DIR = REPO_ROOT / "real_results"
REAL_MODELS_DIR = REAL_RESULTS_DIR / "models"
OUT_DIFFING_DIR = REAL_RESULTS_DIR / "diffing-results"
OUT_PLOTS_DIR = REAL_RESULTS_DIR / "final_plots" / "method2"

BASE_MODEL_SLUGS = ["Qwen__Qwen3-4B", "Qwen__Qwen3.5-4B"]
VARIANTS = ["M1_EM_model_bad_data", "M1_EM_model_good_data"]

VARIANT_LABELS = {
    "M1_EM_model_good_data": r"$\mathbf{EM_{good}}$",
    "M1_EM_model_bad_data": r"$\mathbf{EM_{bad}}$",
}
COLORS = {
    "M1_EM_model_good_data": "tab:green",
    "M1_EM_model_bad_data": "tab:red",
}


def project_per_layer(acts, direction):
    proj = torch.einsum("pld,d->pl", acts.float(), direction)  # [n_prompts, n_layers]
    return proj.mean(dim=0)


def cosine_similarity_per_layer(acts, direction):
    acts_f = acts.float()
    proj = torch.einsum("pld,d->pl", acts_f, direction)
    norms = acts_f.norm(dim=-1).clamp(min=1e-8)
    return (proj / norms).mean(dim=0)


def make_plot(base_cos, entries, direction_layers, out_path):
    n_layers = len(base_cos)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    handles = []
    h, = ax.plot(range(n_layers), base_cos, marker="o", markersize=3, color="tab:blue", label="base")
    handles.append(h)
    for variant, cos in entries:
        h, = ax.plot(range(n_layers), cos, marker="o", markersize=3,
                      color=COLORS[variant], label=VARIANT_LABELS[variant])
        handles.append(h)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for i, l in enumerate(direction_layers[:3]):
        vline = ax.axvline(l, color="green", linestyle=":", alpha=0.7,
                            label=r"$\hat{r}_{base}$" if i == 0 else None)
        if i == 0:
            handles.append(vline)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(-0.25, 0.75)

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(handles), frameon=False, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run():
    OUT_DIFFING_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for slug in BASE_MODEL_SLUGS:
        direction_path = REAL_MODELS_DIR / slug / "M2.1_steer_against_refusal_additive" / "direction.pt"
        if not direction_path.exists():
            print(f"[{slug}] no direction.pt at {direction_path} -- run recompute_direction.py first, skipping")
            continue
        saved = torch.load(direction_path, map_location="cpu")
        direction = saved["direction"].float()
        direction = direction / direction.norm()
        direction_layers = saved["layers"]

        base_acts_path = ACTIVATIONS_DIR / slug / "harmful_val.pt"
        base_acts = torch.load(base_acts_path, map_location="cpu")
        base_proj = project_per_layer(base_acts, direction)
        base_cos = cosine_similarity_per_layer(base_acts, direction)
        n_layers = base_proj.shape[0]

        cos_entries = []
        for variant in VARIANTS:
            tested_dir = ACTIVATIONS_DIR / f"models__{slug}__{variant}"
            tested_acts_path = tested_dir / "harmful_val.pt"
            if not tested_acts_path.exists():
                print(f"[{slug}/{variant}] no cached activations at {tested_acts_path}, skipping")
                continue
            tested_acts = torch.load(tested_acts_path, map_location="cpu")
            tested_proj = project_per_layer(tested_acts, direction)
            tested_cos = cosine_similarity_per_layer(tested_acts, direction)
            cos_entries.append((variant, tested_cos.tolist()))

            result = {
                "method": "projection_on_refusal_direction",
                "tested_model": f"models/{slug.replace('__', '/')}/{variant}",
                "base_model": slug.replace("__", "/"),
                "variant": "M2.1",
                "direction_layers": direction_layers,
                "tested_projection_per_layer": tested_proj.tolist(),
                "base_projection_per_layer": base_proj.tolist(),
                "tested_cosine_per_layer": tested_cos.tolist(),
                "base_cosine_per_layer": base_cos.tolist(),
            }
            json_path = OUT_DIFFING_DIR / f"models__{slug}__{variant}__proj_on_{slug}_M2.1.json"
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[{slug}/{variant}] saved {json_path}")

        if not cos_entries:
            continue
        cos_entries.sort(key=lambda e: e[0])  # bad_data before good_data
        plot_path = OUT_PLOTS_DIR / f"{slug}_cosine.png"
        make_plot(base_cos.tolist(), cos_entries, direction_layers, plot_path)
        print(f"[{slug}] saved {plot_path}")


if __name__ == "__main__":
    run()
