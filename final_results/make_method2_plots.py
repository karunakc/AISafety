"""Re-plot Method 2 (projection on refusal direction) in a cleaner style,
reading ONLY from final_results/diffing-results/*__proj_on_*.json -- no
recomputation, no other data source. Does not touch the original plotting
code in diffing/method2_projection.py; this is a standalone reformatting
script.

For each base model found under final_results/diffing-results/, overlays
the EM_good_data and EM_bad_data variants' cosine-similarity-with-refusal-
direction curves against the shared base-model control on a SINGLE plot
(no raw-projection panel, no figure title):
    final_results/final_plots/method2/<base_model_slug>_cosine.png

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
    "M1_EM_model_good_data": r"$\mathbf{EM_{good}}$",
    "M1_EM_model_bad_data": r"$\mathbf{EM_{bad}}$",
}
COLORS = {
    "M1_EM_model_good_data": "tab:green",
    "M1_EM_model_bad_data": "tab:red",
}

FNAME_RE = re.compile(r"^models__(?P<base_slug>.+)__(?P<variant>M1_EM_model_(?:good|bad)_data)__proj_on_.+\.json$")


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

    by_base = {}
    for json_path in sorted(DIFFING_RESULTS_DIR.glob("*__proj_on_*.json")):
        m = FNAME_RE.match(json_path.name)
        if not m:
            continue
        with open(json_path) as f:
            d = json.load(f)
        by_base.setdefault(m.group("base_slug"), []).append((m.group("variant"), d))

    for base_slug, entries in by_base.items():
        entries.sort(key=lambda e: e[0])  # bad_data before good_data, deterministic order
        out_path = OUT_DIR / f"{base_slug}_cosine.png"
        make_plot(base_slug, entries, out_path)
        print(f"[{base_slug}] wrote {out_path} ({[v for v, _ in entries]})")


if __name__ == "__main__":
    main()
