#!/usr/bin/env python3
"""
Summarize Exp1 COCO gap results from coco_pretrained_gap.py JSON outputs.

Outputs:
- A clean table (Markdown + LaTeX) written to: <out_prefix>_table.txt
- 3 dual-axis scatter plots (PNG + PDF):
    (A) x = I→T R@1
    (B) x = T→I R@1
    (C) x = Avg R@1 = (I→T + T→I)/2
  Left y-axis: Energy gap (black circles + error bars)
  Right y-axis: MMD^2 at σ=median (colored triangles + error bars)

Usage:
  python summarize_exp1_results_v2_fixed.py --result_dir exp1_more_variants --out_prefix exp1_coco --dpi 300
"""
from __future__ import annotations

import os
import glob
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_jsons(result_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        if os.path.basename(fp) == "index.json":
            continue
        with open(fp, "r") as f:
            d = json.load(f)
        d["_file"] = os.path.basename(fp)
        rows.append(d)
    return rows


def pm(mean: Optional[float], std: Optional[float], digits: int = 3) -> str:
    if mean is None or std is None:
        return ""
    if np.isnan(mean) or np.isnan(std):
        return ""
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def _find_mmd_key_by_sigma(row: Dict[str, Any], sigma: float, suffix: str) -> Optional[str]:
    """Return key like 'mmd2_sigma=1.2949_mean' using 4-dec rounding."""
    k = f"mmd2_sigma={sigma:.4f}_{suffix}"
    return k if k in row else None


def _find_closest_mmd_key(row: Dict[str, Any], target_sigma: float, suffix: str) -> Optional[str]:
    """Fallback: parse all mmd2_sigma=..._{suffix} keys and choose closest sigma."""
    best_k = None
    best_diff = None
    for k in row.keys():
        if not k.startswith("mmd2_sigma=") or not k.endswith(f"_{suffix}"):
            continue
        try:
            mid = k[len("mmd2_sigma="):].split("_")[0]
            s = float(mid)
        except Exception:
            continue
        diff = abs(s - target_sigma)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_k = k
    return best_k


def get_mmd_at_median_sigma(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (mmd_mean, mmd_std, sigma_med_used).
    Prefers sigma = sigmas[1] (1x median heuristic). Otherwise uses sigma_1.0x or median_cross_dist.
    """
    sigma_med = None
    if isinstance(row.get("sigmas", None), list) and len(row["sigmas"]) >= 2:
        sigma_med = float(row["sigmas"][1])
    elif row.get("sigma_1.0x", None) is not None:
        sigma_med = float(row["sigma_1.0x"])
    elif row.get("median_cross_dist", None) is not None:
        sigma_med = float(row["median_cross_dist"])

    if sigma_med is None:
        return None, None, None

    k_mean = _find_mmd_key_by_sigma(row, sigma_med, "mean") or _find_closest_mmd_key(row, sigma_med, "mean")
    k_std  = _find_mmd_key_by_sigma(row, sigma_med, "std")  or _find_closest_mmd_key(row, sigma_med, "std")

    if k_mean is None or k_std is None:
        return None, None, sigma_med

    try:
        return float(row[k_mean]), float(row[k_std]), sigma_med
    except Exception:
        return None, None, sigma_med


def build_table_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        mmd_mean, mmd_std, s_med = get_mmd_at_median_sigma(r)
        i2t = r.get("i2t_R@1", None)
        t2i = r.get("t2i_R@1", None)

        i2t_f = float(i2t) if i2t is not None else None
        t2i_f = float(t2i) if t2i is not None else None
        avg_f = None if (i2t_f is None or t2i_f is None) else 0.5 * (i2t_f + t2i_f)

        out.append({
            "Model": r.get("model_name", ""),
            "Pretrained": r.get("pretrained", ""),
            "I2T_R1": i2t_f,
            "T2I_R1": t2i_f,
            "AVG_R1": avg_f,
            "Energy_mean": float(r.get("energy_mean", np.nan)),
            "Energy_std": float(r.get("energy_std", np.nan)),
            "MMD_med_mean": float(mmd_mean) if mmd_mean is not None else np.nan,
            "MMD_med_std": float(mmd_std) if mmd_std is not None else np.nan,
            "sigma_med": float(s_med) if s_med is not None else np.nan,
            "Centroid_gap": float(r.get("centroid_gap", np.nan)),
            "mu_cos": float(r.get("mu_cosine", np.nan)),
            "_file": r.get("_file", ""),
        })
    return out


def write_tables(table_rows: List[Dict[str, Any]], out_txt: str, digits: int = 3):
    # Sort by AVG_R1 desc if available else I2T_R1
    def sort_key(x):
        if x["AVG_R1"] is not None and not np.isnan(x["AVG_R1"]):
            return -x["AVG_R1"]
        if x["I2T_R1"] is not None and not np.isnan(x["I2T_R1"]):
            return -x["I2T_R1"]
        return 0.0

    rows = sorted(table_rows, key=sort_key)

    # Markdown table
    cols = ["Model", "I→T R@1", "T→I R@1", "Energy", "MMD²@σ=med", "Controd Gap (||μI-μT||)", "Mean Similarity (cos(μI,μT))"]
    md = []
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        md.append("| " + " | ".join([
            r["Model"],
            "" if r["I2T_R1"] is None else f"{r['I2T_R1']:.{digits}f}",
            "" if r["T2I_R1"] is None else f"{r['T2I_R1']:.{digits}f}",
            pm(r["Energy_mean"], r["Energy_std"], digits),
            pm(r["MMD_med_mean"], r["MMD_med_std"], digits),
            "" if np.isnan(r["Centroid_gap"]) else f"{r['Centroid_gap']:.{digits}f}",
            "" if np.isnan(r["mu_cos"]) else f"{r['mu_cos']:.{digits}f}",
        ]) + " |")

    # LaTeX tabular (avoid nested f-strings)
    latex = []
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & I$\to$T R@1 & T$\to$I R@1 & Energy $\downarrow$ & MMD$^2_{\sigma=\mathrm{med}} \downarrow$ & $\|\mu_I-\mu_T\|$ & $\cos(\mu_I,\mu_T)$ \\")
    latex.append(r"\midrule")
    for r in rows:
        i2t_str = "" if r["I2T_R1"] is None else f"{r['I2T_R1']:.{digits}f}"
        t2i_str = "" if r["T2I_R1"] is None else f"{r['T2I_R1']:.{digits}f}"
        energy_str = pm(r["Energy_mean"], r["Energy_std"], digits)
        mmd_str = pm(r["MMD_med_mean"], r["MMD_med_std"], digits)
        cent_str = "" if np.isnan(r["Centroid_gap"]) else f"{r['Centroid_gap']:.{digits}f}"
        cos_str = "" if np.isnan(r["mu_cos"]) else f"{r['mu_cos']:.{digits}f}"
        latex.append(f"{r['Model']} & {i2t_str} & {t2i_str} & {energy_str} & {mmd_str} & {cent_str} & {cos_str} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")

    with open(out_txt, "w") as f:
        f.write("Exp1 summary table (Markdown)\n")
        f.write("============================\n\n")
        f.write("\n".join(md) + "\n\n")
        f.write("Exp1 summary table (LaTeX tabular)\n")
        f.write("=================================\n\n")
        f.write("\n".join(latex) + "\n")

    print(f"Wrote table txt: {out_txt}")


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
})

def _setup_axes(ax, ax2, x_label: str):
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy gap")
    ax2.set_ylabel("Centroid gap")  # <-- changed

    # Left axis (Energy): Stronger, dashed grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.5)
    # Right axis (Centroid): Lighter, dotted grid
    ax2.grid(True, which="major", linestyle=":", linewidth=0.6, color="gray", alpha=0.3)

    # Black bounding box (spines)
    for axis in [ax, ax2]:
        for spine in axis.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
            spine.set_visible(True)

    # Inward ticks
    ax.tick_params(direction="in", length=5, width=0.8, colors="black", top=True, right=False)
    ax2.tick_params(direction="in", length=5, width=0.8, colors="black", left=False, right=True)


def _build_model_color_map(table_rows: List[Dict[str, Any]], cmap_name: str = "tab10") -> Dict[str, Any]:
    models = sorted({r.get("Model", "") for r in table_rows if r.get("Model", "")})
    cmap = plt.get_cmap(cmap_name)
    n = max(1, len(models))
    return {m: cmap(i % cmap.N) for i, m in enumerate(models)}

def dual_axis_scatter(
    table_rows: List[Dict[str, Any]],
    x_key: str,
    x_label: str,
    out_prefix: str,
    annotate: bool = True,
    dpi: int = 300,
    markersize: int = 8,
    capsize: int = 3,
    linewidth: float = 1.2,
    cmap_name: str = "tab10",
):
    color_map = _build_model_color_map(table_rows, cmap_name=cmap_name)

    rows = [r for r in table_rows if r.get(x_key, None) is not None and not np.isnan(r[x_key])]
    # changed filter: require Centroid_gap instead of MMD
    rows = [r for r in rows if not np.isnan(r["Energy_mean"]) and not np.isnan(r["Centroid_gap"])]

    if len(rows) == 0:
        print(f"[warn] No valid rows for x_key={x_key}")
        return

    rows = sorted(rows, key=lambda r: float(r[x_key]))

    xs = np.array([float(r[x_key]) for r in rows], dtype=float)
    e = np.array([float(r["Energy_mean"]) for r in rows], dtype=float)
    e_std = np.array([float(r["Energy_std"]) for r in rows], dtype=float)

    # right axis now uses centroid gap (no std available in your JSON)
    c = np.array([float(r["Centroid_gap"]) for r in rows], dtype=float)
    c_std = np.zeros_like(c)  # keeps errorbar call signature; shows no uncertainty

    labels = [r["Model"] for r in rows]
    colors = [color_map.get(lab, (0.2, 0.2, 0.2, 1.0)) for lab in labels]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax2 = ax.twinx()

    _setup_axes(ax, ax2, x_label=x_label)

    for i in range(len(rows)):
        col = colors[i]

        # Energy on left axis
        ax.errorbar(
            xs[i], e[i], yerr=e_std[i],
            fmt="o",
            markersize=markersize,
            linewidth=linewidth,
            capsize=capsize,
            color=col,
            ecolor=col,
            zorder=3,
            alpha=0.9
        )

        # Centroid gap on right axis (no error bar; c_std=0)
        ax2.errorbar(
            xs[i], c[i], yerr=c_std[i],
            fmt=">",
            markersize=markersize,
            linewidth=linewidth,
            capsize=capsize,
            color=col,
            ecolor=col,
            zorder=3,
            alpha=0.5
        )

    # Legends (keep errorbar-style handles)
    proxy_energy = ax.errorbar(
        [np.nan], [np.nan], yerr=[np.nan],
        fmt="o", markersize=markersize,
        linewidth=linewidth, capsize=capsize, 
        color="black", ecolor="black",
        label="Energy gap (left axis)",
        alpha=0.9
    )

    # For centroid, we show an errorbar glyph in legend for style consistency only.
    # (Centroid std is not estimated in your JSON; plotted points have yerr=0.)
    proxy_centroid = ax2.errorbar(
        [np.nan], [np.nan], yerr=[1.0],   # legend glyph only; doesn't affect plot scaling
        fmt=">", markersize=markersize,
        linewidth=linewidth, capsize=capsize,
        color="black", ecolor="black",
        label="Centroid gap (right axis)",
        alpha=0.5
    )

    ax.legend(
        handles=[proxy_energy, proxy_centroid],
        loc="best",
        frameon=True,
        edgecolor="gray",
        fancybox=False,
        framealpha=0.7,
        fontsize=9
    )

    if annotate:
        x_min, x_max = ax.get_xlim()
        x_span = max(1e-9, x_max - x_min)
        y_min, y_max = ax.get_ylim()
        y_span = max(1e-9, y_max - y_min)
        
        right_zone = x_max - 0.5 * x_span
        bottom_zone = y_min + 0.5 * y_span
        
        for i in range(len(rows)):
            col = colors[i]
            x = xs[i]
            y = c[i]  # <-- annotate centroid points

            if x >= right_zone:
                dx, ha = -8, "right"
            else:
                dx, ha = 8, "left"
            
            if y <= bottom_zone:
                dy = 10
            else:
                dy = -10

            ax2.annotate(
                labels[i],
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va="bottom",
                fontsize=14,
                color=col,
                annotation_clip=True,
                bbox=dict(
                    boxstyle="square,pad=0.2",
                    fc="none",
                    ec="none",
                    lw=0.5,
                    alpha=0.9,
                ),
            )

    plt.tight_layout()
    out_png = f"{out_prefix}_{x_key}_dual.png"
    out_pdf = f"{out_prefix}_{x_key}_dual.pdf"
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    plt.close(fig)
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="exp1")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--no_annotate", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    raw = load_jsons(args.result_dir)
    table_rows = build_table_rows(raw)

    # Aesthetic style (matplotlib style only; no seaborn dependency)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Write table txt (Markdown + LaTeX)
    out_txt = f"{args.out_prefix}_table.txt"
    write_tables(table_rows, out_txt=out_txt, digits=args.digits)

    # Dual-axis plots requested
    dual_axis_scatter(
        table_rows, x_key="I2T_R1",
        x_label="I→T Retrieval R@1",
        out_prefix=args.out_prefix,
        annotate=(not args.no_annotate),
        dpi=args.dpi,
    )
    dual_axis_scatter(
        table_rows, x_key="T2I_R1",
        x_label="T→I Retrieval R@1",
        out_prefix=args.out_prefix,
        annotate=(not args.no_annotate),
        dpi=args.dpi,
    )
    dual_axis_scatter(
        table_rows, x_key="AVG_R1",
        x_label="Average Retrieval R@1",
        out_prefix=args.out_prefix,
        annotate=(not args.no_annotate),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
    """
    python exp1_summarize_results.py --result_dir exp1_results --out_prefix exp1_summary/coco_pretrained --dpi 600
    """