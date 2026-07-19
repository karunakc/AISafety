"""Re-plot Method 2 (projection on refusal direction) in a cleaner style,
reading ONLY from final_results/diffing-results/*__proj_on_*.json -- no
recomputation, no other data source. Does not touch the original plotting
code in diffing/method2_projection.py; this is a standalone reformatting
script.

For each base model found under final_results/diffing-results/, produces
THREE plots (no raw-projection panel, no figure title):
    final_results/final_plots/method2/<base_model_slug>_cosine.png
        base + EM_good_data (r=32) + EM_bad_data (r=32) -- original combined view
    final_results/final_plots/method2/<base_model_slug>_good_cosine.png
        base + EM_good_data (r=32) + EM_good_data_low_lora (r=8)
    final_results/final_plots/method2/<base_model_slug>_bad_cosine.png
        base + EM_bad_data (r=32) + EM_bad_data_low_lora (r=8)

Usage:
    python final_results/make_method2_plots.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})

FINAL_RESULTS_DIR = Path(__file__).resolve().parent
DIFFING_RESULTS_DIR = FINAL_RESULTS_DIR / "diffing-results"
OUT_DIR = FINAL_RESULTS_DIR / "final_plots" / "method2"

VARIANT_LABELS = {
    "M1_EM_model_good_data": r"$\mathbf{EM_{good}}$ (r=32)",
    "M1_EM_model_bad_data": r"$\mathbf{EM_{bad}}$ (r=32)",
    "M1_EM_model_good_data_low_lora": r"$\mathbf{EM_{good}}$ (r=8)",
    "M1_EM_model_bad_data_low_lora": r"$\mathbf{EM_{bad}}$ (r=8)",
}
COLORS = {
    "M1_EM_model_good_data": "tab:green",
    "M1_EM_model_bad_data": "tab:red",
    "M1_EM_model_good_data_low_lora": "tab:purple",
    "M1_EM_model_bad_data_low_lora": "tab:orange",
}

FNAME_RE = re.compile(
    r"^models__(?P<base_slug>.+)__(?P<variant>M1_EM_model_(?:good|bad)_data(?:_low_lora)?)__proj_on_.+\.json$"
)
CATEGORY_OF = {
    "M1_EM_model_good_data": "good",
    "M1_EM_model_good_data_low_lora": "good",
    "M1_EM_model_bad_data": "bad",
    "M1_EM_model_bad_data_low_lora": "bad",
}
# within a category, plot the full-rank (r=32) curve before the low-rank one
VARIANT_ORDER = {
    "M1_EM_model_good_data": 0,
    "M1_EM_model_bad_data": 0,
    "M1_EM_model_good_data_low_lora": 1,
    "M1_EM_model_bad_data_low_lora": 1,
}


def make_plot(base_slug, entries, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    base_cos = entries[0][1]["base_cosine_per_layer"]
    n_layers = len(base_cos)
    direction_layers = entries[0][1]["direction_layers"]

    handles = []
    h, = ax.plot(range(n_layers), base_cos, marker="o", markersize=3, color="tab:blue", label="base")
    handles.append(h)
    for variant, d in entries:
        h, = ax.plot(range(n_layers), d["tested_cosine_per_layer"], marker="o", markersize=3,
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_base_category = {}
    by_base_combined = {}
    for json_path in sorted(DIFFING_RESULTS_DIR.glob("*__proj_on_*.json")):
        m = FNAME_RE.match(json_path.name)
        if not m:
            continue
        with open(json_path) as f:
            d = json.load(f)
        variant = m.group("variant")
        base_slug = m.group("base_slug")
        key = (base_slug, CATEGORY_OF[variant])
        by_base_category.setdefault(key, []).append((variant, d))
        if not variant.endswith("_low_lora"):  # r=32 only, original combined view
            by_base_combined.setdefault(base_slug, []).append((variant, d))

    for base_slug, entries in by_base_combined.items():
        entries.sort(key=lambda e: e[0])  # bad_data before good_data, deterministic order
        out_path = OUT_DIR / f"{base_slug}_cosine.png"
        make_plot(base_slug, entries, out_path)
        print(f"[{base_slug}] wrote {out_path} ({[v for v, _ in entries]})")

    for (base_slug, category), entries in by_base_category.items():
        entries.sort(key=lambda e: VARIANT_ORDER[e[0]])
        out_path = OUT_DIR / f"{base_slug}_{category}_cosine.png"
        make_plot(base_slug, entries, out_path)
        print(f"[{base_slug}/{category}] wrote {out_path} ({[v for v, _ in entries]})")


if __name__ == "__main__":
    main()
