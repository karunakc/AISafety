"""Re-plot Method 5 (steered refusal-metric distribution) in a cleaner style,
reading ONLY from final_results/diffing-results/*__steered_dist_*.json -- no
recomputation, no other data source. Does not touch the original plotting
code in diffing/method5_steered_distribution.py; this is a standalone
reformatting script.

For each steered_dist JSON found under final_results/diffing-results/,
produces one file under final_results/final_plots/diffing_method_5/:
    <same stem>.png -- two panels side by side, no figure title:
        left:  clean harmful/harmless distribution, titled "Clean"
        right: steered harmful/harmless distribution, titled "Steered"

Usage:
    python final_results/make_method5_plots.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

FINAL_RESULTS_DIR = Path(__file__).resolve().parent
DIFFING_RESULTS_DIR = FINAL_RESULTS_DIR / "diffing-results"
OUT_DIR = FINAL_RESULTS_DIR / "final_plots" / "diffing_method_5"


def make_plot(d, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    ax1.hist(d["harmless_clean"], bins=30, color="tab:blue", alpha=0.6, label="harmless")
    ax1.hist(d["harmful_clean"], bins=30, color="tab:red", alpha=0.6, label="harmful")
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Refusal metric")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Clean")
    ax1.legend()

    ax2.hist(d["harmless_steered"], bins=30, color="tab:blue", alpha=0.6, label="harmless")
    ax2.hist(d["harmful_steered"], bins=30, color="tab:red", alpha=0.6, label="harmful")
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Refusal metric")
    ax2.set_title("Steered")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for json_path in sorted(DIFFING_RESULTS_DIR.glob("*__steered_dist_*.json")):
        with open(json_path) as f:
            d = json.load(f)
        out_path = OUT_DIR / f"{json_path.stem}.png"
        make_plot(d, out_path)
        print(f"[{json_path.stem}] wrote {out_path}")


if __name__ == "__main__":
    main()
