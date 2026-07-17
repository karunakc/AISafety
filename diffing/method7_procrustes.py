"""
Method 7: Orthogonal Procrustes alignment between Base and LoRA representation
spaces (representational analysis only -- does NOT touch steering/behavior;
that is a separate follow-up experiment once this one's results support it).

Motivation (from methods 5/6): global hidden representations stay highly
similar after LoRA fine-tuning (high CKA), but the refusal direction itself
rotates substantially, especially for the Risky LoRA. This asks: is that
rotation explained by a SINGLE per-layer orthogonal transformation of the
whole representation space -- i.e. is the refusal concept preserved but
expressed in a rotated coordinate system, or has LoRA genuinely changed what
the refusal direction means?

For every layer, with H_base, H_lora in R^{N x d} (identical prompts, same
rows, mean-centered per feature -- same centering convention as
method5_cka.py's CKA and the same combined harmful+harmless activation set
its "all" curve uses):

    R = argmin_{R^T R = I} ||H_base @ R - H_lora||_F

the standard orthogonal Procrustes solution (Schönemann 1966): form
M = H_base^T @ H_lora (d x d), take its full SVD M = U S V^T, and R = U V^T.

Since R was fit to solve H_base @ R ≈ H_lora (R acts on the right of each
row/sample), a single direction vector is aligned the same way: r_aligned =
r_base @ R, not R @ r_base. Then r_aligned is compared against r_lora, both before and
after alignment (cosine similarity, angular distance in degrees), plus the
raw Frobenius reconstruction error ||H_base @ R - H_lora||_F vs.
||H_base - H_lora||_F as a companion diagnostic of how well a pure rotation
explains the representational gap.

d (3584+) is large enough that this SVD is real compute (a full d x d SVD
takes ~20s on CPU per layer -- run this on a GPU, not the login node/a bare
CPU loop, e.g. via the diffing_method7 Modal entrypoint).

Also computes a GENERAL linear map (no orthogonality constraint) as a
comparison point, per the standard three-way read: if orthogonal Procrustes
already recovers r_lora well, the change is mostly geometric (a rotation);
if only the unconstrained map works noticeably better, LoRA introduced
scaling/shearing beyond a pure rotation; if neither helps, the refusal
concept itself changed. IMPORTANT CAVEAT: N (64 prompts) << d (2560+), so
BOTH fits are heavily underdetermined -- the unconstrained map in particular
has d^2 free parameters against N*d constraints and can drive the Frobenius
error to ~0 almost by construction, which is expected overfitting to reach
for, not evidence of a "richer" transformation. Read the unconstrained
numbers as an overfitting upper bound the orthogonal result is compared
against, not as a second independently trustworthy alignment.

Reuses the SAME activation cache diffing/method5_cka.py's
get_or_compute_activations already produced (data/refusal/activations/
<base_slug>__<variant>/{harmful,harmless}_<split>.pt) -- raises if missing
rather than computing live, since (like method6) this is meant to run as a
free-standing diagnostic on activations another run already paid the GPU
cost for.

Usage:
    python diffing/method7_procrustes.py --base_model Qwen/Qwen3.5-4B \\
        --variants M1_good_medical_advice,M1_risky_financial_advice
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import model_slug  # noqa: E402
from refusal_misaligned import load_activations  # noqa: E402
from method6_angular_distance import (  # noqa: E402
    DEFAULT_VARIANTS,
    angular_distance_per_layer,
    display_name,
    load_refusal_directions,
    resolve_activations_dir,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Fixed categorical order (never cycled/reassigned) -- same slots
# diffing/method5_cka.py's "all"/"harmful"/"harmless" curves use.
RAW_COLOR = "#2a78d6"
ORTHOGONAL_COLOR = "#1baf7a"
GENERAL_COLOR = "#eda100"


def load_combined_activations(base_model, variant, split):
    """[N, n_layers, d] = concat(harmful, harmless) along N -- the exact
    same combined activation set diffing/method5_cka.py's "all" CKA curve
    uses, reused here (no recomputation)."""
    acts_dir = resolve_activations_dir(base_model, variant)
    if not (acts_dir / f"harmful_{split}.pt").exists():
        raise FileNotFoundError(
            f"No cached activations at {acts_dir} for split={split!r}. "
            f"Run diffing/method5_cka.py for {base_model} [{variant}] first."
        )
    harmful, harmless = load_activations(acts_dir, split)
    return torch.cat([harmful, harmless], dim=0)


def orthogonal_procrustes(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A, B: (N, d), already centered. Returns R: (d, d) orthogonal, the
    standard SVD solution to argmin_{R^T R = I} ||A R - B||_F."""
    M = A.T @ B
    U, _, Vh = torch.linalg.svd(M, full_matrices=True)
    return U @ Vh


def general_linear_map(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A, B: (N, d), already centered. Returns W: (d, d), the ordinary
    least-squares (minimum-norm, via Moore-Penrose pseudoinverse) solution
    to argmin_W ||A W - B||_F -- NO orthogonality constraint. Since N << d
    here, this is heavily underdetermined and can reach near-zero Frobenius
    error almost by construction (see module docstring); it's a comparison
    upper bound for orthogonal_procrustes, not a second trustworthy alignment
    on its own."""
    return torch.linalg.pinv(A) @ B


def run_variant(H_base, H_lora, r_base):
    """H_base/H_lora: [N, n_layers, d]. r_base: [n_layers, d] (unit-normalized).
    Returns a dict of stacked per-layer results for both the orthogonal and
    general-linear-map alignments."""
    n_layers = H_base.shape[1]
    out = {k: [] for k in (
        "R", "W", "aligned_orthogonal", "aligned_general",
        "frob_before", "frob_after_orthogonal", "frob_after_general",
        "general_map_norm_ratio",
    )}
    for L in range(n_layers):
        A = H_base[:, L, :].float()
        B = H_lora[:, L, :].float()
        A_c = A - A.mean(dim=0, keepdim=True)
        B_c = B - B.mean(dim=0, keepdim=True)

        R = orthogonal_procrustes(A_c, B_c)
        W = general_linear_map(A_c, B_c)
        # Both R and W solve argmin ||A @ T - B||_F (row-vector convention:
        # T acts on the right of each sample row) -- so a single direction
        # vector must be aligned the same way, r_base @ T, not T @ r_base.
        r_base_L = r_base[L].float()
        r_orth = r_base_L @ R
        r_orth = r_orth / r_orth.norm().clamp(min=1e-12)
        r_gen_raw = r_base_L @ W
        r_gen_norm = r_gen_raw.norm().item()  # far from 1.0 flags a degenerate/overfit map
        r_gen = r_gen_raw / max(r_gen_norm, 1e-12)

        out["R"].append(R)
        out["W"].append(W)
        out["aligned_orthogonal"].append(r_orth)
        out["aligned_general"].append(r_gen)
        out["frob_before"].append((A_c - B_c).norm().item())
        out["frob_after_orthogonal"].append((A_c @ R - B_c).norm().item())
        out["frob_after_general"].append((A_c @ W - B_c).norm().item())
        out["general_map_norm_ratio"].append(r_gen_norm)
        print(f"  layer {L:3d}: frobenius {out['frob_before'][-1]:.2f} -> "
              f"orthogonal {out['frob_after_orthogonal'][-1]:.2f}, general {out['frob_after_general'][-1]:.2f}")

    out["R"] = torch.stack(out["R"])
    out["W"] = torch.stack(out["W"])
    out["aligned_orthogonal"] = torch.stack(out["aligned_orthogonal"])
    out["aligned_general"] = torch.stack(out["aligned_general"])
    return out


def run(base_model, variants=None, split="val", label=None, output_dir=None):
    variants = variants or DEFAULT_VARIANTS
    results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    out_stem = label or f"{model_slug(base_model)}__procrustes"

    r_base_all = load_refusal_directions(base_model, "base", split)  # [n_layers, d]
    n_layers = r_base_all.shape[0]

    summary = {}
    for variant in variants:
        print(f"=== {variant} ===")
        r_lora = load_refusal_directions(base_model, variant, split)
        H_base = load_combined_activations(base_model, "base", split)
        H_lora = load_combined_activations(base_model, variant, split)
        if H_base.shape[1] != n_layers or H_lora.shape[1] != n_layers:
            raise ValueError(f"Layer count mismatch for {variant}.")

        res = run_variant(H_base, H_lora, r_base_all)

        cosine_before, angle_before = angular_distance_per_layer(r_base_all, r_lora)
        cosine_orth, angle_orth = angular_distance_per_layer(res["aligned_orthogonal"], r_lora)
        cosine_gen, angle_gen = angular_distance_per_layer(res["aligned_general"], r_lora)

        transforms_path = results_dir / f"{out_stem}_{variant}_transforms.pt"
        torch.save({
            "R": res["R"], "W": res["W"], "r_base": r_base_all, "r_lora": r_lora,
            "r_aligned_orthogonal": res["aligned_orthogonal"], "r_aligned_general": res["aligned_general"],
        }, transforms_path)
        print(f"Saved transformation matrices + aligned vectors to {transforms_path}")

        mean_cos_before, mean_cos_orth, mean_cos_gen = float(cosine_before.mean()), float(cosine_orth.mean()), float(cosine_gen.mean())
        mean_angle_before, mean_angle_orth, mean_angle_gen = float(angle_before.mean()), float(angle_orth.mean()), float(angle_gen.mean())
        mean_norm_ratio = sum(res["general_map_norm_ratio"]) / n_layers
        print(f"[{variant}] mean cosine: {mean_cos_before:.4f} -> orthogonal {mean_cos_orth:.4f}, general {mean_cos_gen:.4f}  "
              f"(mean angle: {mean_angle_before:.2f} -> orthogonal {mean_angle_orth:.2f}, general {mean_angle_gen:.2f} deg)  "
              f"[general map ||r_base @ W|| avg {mean_norm_ratio:.3f} before renormalization -- far from 1.0 flags overfitting]")

        summary[variant] = {
            "cosine_before": cosine_before.tolist(),
            "cosine_after_orthogonal": cosine_orth.tolist(),
            "cosine_after_general": cosine_gen.tolist(),
            "angle_deg_before": angle_before.tolist(),
            "angle_deg_after_orthogonal": angle_orth.tolist(),
            "angle_deg_after_general": angle_gen.tolist(),
            "improvement_cosine_orthogonal": (cosine_orth - cosine_before).tolist(),
            "improvement_cosine_general": (cosine_gen - cosine_before).tolist(),
            "improvement_angle_deg_orthogonal": (angle_before - angle_orth).tolist(),
            "improvement_angle_deg_general": (angle_before - angle_gen).tolist(),
            "frobenius_before": res["frob_before"],
            "frobenius_after_orthogonal": res["frob_after_orthogonal"],
            "frobenius_after_general": res["frob_after_general"],
            "general_map_norm_ratio": res["general_map_norm_ratio"],
            "transforms_path": str(transforms_path),
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(range(n_layers), cosine_before, marker="o", markersize=3, label="raw", color=RAW_COLOR)
        ax1.plot(range(n_layers), cosine_orth, marker="o", markersize=3, label="orthogonal Procrustes", color=ORTHOGONAL_COLOR)
        ax1.plot(range(n_layers), cosine_gen, marker="o", markersize=3, label="general linear map", color=GENERAL_COLOR)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Cosine similarity with LoRA refusal direction")
        ax1.set_ylim(-1, 1.02)
        ax1.set_title("Cosine similarity")
        ax1.legend()

        ax2.plot(range(n_layers), angle_before, marker="o", markersize=3, label="raw", color=RAW_COLOR)
        ax2.plot(range(n_layers), angle_orth, marker="o", markersize=3, label="orthogonal Procrustes", color=ORTHOGONAL_COLOR)
        ax2.plot(range(n_layers), angle_gen, marker="o", markersize=3, label="general linear map", color=GENERAL_COLOR)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Angular distance (degrees)")
        ax2.set_ylim(0, 90)
        ax2.set_title("Angular distance")
        ax2.legend()

        fig.suptitle(f"Procrustes alignment: Base → {display_name(variant)}")
        plt.tight_layout()
        plot_path = results_dir / f"{out_stem}_{variant}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")

    json_path = results_dir / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump({
            "method": "orthogonal_procrustes_per_layer",
            "base_model": base_model,
            "variants": variants,
            "split": split,
            "results": summary,
        }, f, indent=2)
    print(f"Saved result to {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Method 7: per-layer orthogonal Procrustes alignment, Base -> LoRA.")
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--label", default=None, help="Output filename stem under diffing/results/ (default: auto-generated)")
    parser.add_argument("--output_dir", default=None, help="Directory to save results to (default: diffing/results/)")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run(args.base_model, variants, args.split, args.label, args.output_dir)


if __name__ == "__main__":
    main()
