#!/usr/bin/env python3

from __future__ import annotations
import os, glob, json, re, argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parsing helpers
# -------------------------
FNAME_RE = re.compile(r".*exp2_(?P<model>.+?)_p(?P<p>[0-9]+(?:\.[0-9]+)?)_eval\.json$")

def parse_model_p_from_filename(fp: str) -> Tuple[Optional[str], Optional[float]]:
    m = FNAME_RE.match(os.path.basename(fp))
    if not m:
        return None, None
    model = m.group("model")
    p = float(m.group("p"))
    return model, p

def load_jsons(result_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        if os.path.basename(fp) == "index.json":
            continue
        model, p = parse_model_p_from_filename(fp)
        if model is None or p is None:
            # Skip files not matching exp2 naming convention.
            continue
        with open(fp, "r") as f:
            d = json.load(f)
        d["_file"] = os.path.basename(fp)
        d["_model_from_name"] = model
        d["_p"] = p
        rows.append(d)
    return rows

def pm(mean: float, std: float, digits: int = 3) -> str:
    if mean is None or std is None:
        return ""
    if np.isnan(mean) or np.isnan(std):
        return ""
    return f"{mean:.{digits}f}±{std:.{digits}f}"

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# -------------------------
# Build aggregated rows
# -------------------------
def build_groups(raw_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, float], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        model = r.get("model_name", "") or r.get("_model_from_name", "")
        # prefer filename model because you rely on that naming
        model = r.get("_model_from_name", model)
        p = float(r.get("_p"))
        key = (model, p)
        groups.setdefault(key, []).append(r)
    return groups

def aggregate_groups(groups: Dict[Tuple[str, float], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Each output row corresponds to one (model, p), aggregating repeated runs.
    """
    out: List[Dict[str, Any]] = []
    for (model, p), runs in groups.items():
        i2t = np.array([safe_float(r.get("i2t_R@1", np.nan)) for r in runs], dtype=np.float64)
        t2i = np.array([safe_float(r.get("t2i_R@1", np.nan)) for r in runs], dtype=np.float64)
        cent = np.array([safe_float(r.get("centroid_gap", np.nan)) for r in runs], dtype=np.float64)

        # avg retrieval per-run then aggregate (avoids bias if some run missing one side)
        avg = []
        for a, b in zip(i2t, t2i):
            if np.isnan(a) or np.isnan(b):
                avg.append(np.nan)
            else:
                avg.append(0.5 * (a + b))
        avg = np.array(avg, dtype=np.float64)

        def mean_std(v: np.ndarray) -> Tuple[float, float, int]:
            v2 = v[~np.isnan(v)]
            if len(v2) == 0:
                return np.nan, np.nan, 0
            if len(v2) == 1:
                return float(v2[0]), 0.0, 1
            return float(v2.mean()), float(v2.std(ddof=1)), int(len(v2))

        i2t_m, i2t_s, n1 = mean_std(i2t)
        t2i_m, t2i_s, n2 = mean_std(t2i)
        avg_m, avg_s, n3 = mean_std(avg)
        cen_m, cen_s, n4 = mean_std(cent)

        out.append({
            "Model": model,
            "p": float(p),
            "n_runs": int(max(n1, n2, n3, n4)),
            "I2T_mean": i2t_m, "I2T_std": i2t_s,
            "T2I_mean": t2i_m, "T2I_std": t2i_s,
            "AVG_mean": avg_m, "AVG_std": avg_s,
            "Centroid_mean": cen_m, "Centroid_std": cen_s,
        })

    # Sort: model then p
    out.sort(key=lambda r: (r["Model"], r["p"]))
    return out

# -------------------------
# Tables
# -------------------------
def write_tables(rows: List[Dict[str, Any]], out_txt: str, digits: int = 3):
    # Combined markdown table
    md = []
    cols = ["Model", "p", "Avg R@1", "I→T R@1", "T→I R@1", "Centroid gap ||μI-μT||", "n"]
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")

    for r in rows:
        md.append("| " + " | ".join([
            str(r["Model"]),
            f"{r['p']:.2f}",
            pm(r["AVG_mean"], r["AVG_std"], digits),
            pm(r["I2T_mean"], r["I2T_std"], digits),
            pm(r["T2I_mean"], r["T2I_std"], digits),
            pm(r["Centroid_mean"], r["Centroid_std"], digits),
            str(r["n_runs"]),
        ]) + " |")

    # LaTeX tabular
    latex = []
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & $p$ & Avg R@1 & I$\to$T R@1 & T$\to$I R@1 & $\|\mu_I-\mu_T\|$ & $n$ \\")
    latex.append(r"\midrule")
    for r in rows:
        latex.append(
            f"{r['Model']} & {r['p']:.2f} & "
            f"{pm(r['AVG_mean'], r['AVG_std'], digits)} & "
            f"{pm(r['I2T_mean'], r['I2T_std'], digits)} & "
            f"{pm(r['T2I_mean'], r['T2I_std'], digits)} & "
            f"{pm(r['Centroid_mean'], r['Centroid_std'], digits)} & "
            f"{r['n_runs']} \\\\"
        )
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")

    with open(out_txt, "w") as f:
        f.write("Exp2 summary table (Markdown)\n")
        f.write("============================\n\n")
        f.write("\n".join(md) + "\n\n")
        f.write("Exp2 summary table (LaTeX tabular)\n")
        f.write("=================================\n\n")
        f.write("\n".join(latex) + "\n")

    print(f"Wrote table txt: {out_txt}")

# -------------------------
# Plot style helpers (similar to your Exp1)
# -------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
})

def _setup_axes(ax, ax2, x_label: str, left_label: str, right_label: str, grid: bool):
    ax.set_xlabel(x_label)
    ax.set_ylabel(left_label)
    ax2.set_ylabel(right_label)

    # Grids (toggle)
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.5)
        ax2.grid(True, which="major", linestyle=":", linewidth=0.6, color="gray", alpha=0.3)
    else:
        ax.grid(False)
        ax2.grid(False)

    # Black spines
    for axis in (ax, ax2):
        for spine in axis.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
            spine.set_visible(True)

    # Inward ticks
    ax.tick_params(direction="in", length=5, width=0.8, colors="black", top=True, right=False)
    ax2.tick_params(direction="in", length=5, width=0.8, colors="black", left=False, right=True)

def _build_model_color_map(models: List[str], cmap_name: str = "tab10") -> Dict[str, Any]:
    cmap = plt.get_cmap(cmap_name)
    return {m: cmap(i % cmap.N) for i, m in enumerate(models)}


def plot_exp2_curve_and_bar(
    rows: List[Dict[str, Any]],
    retrieval_mean_key: str,
    retrieval_std_key: str,
    retrieval_label: str,
    out_prefix: str,
    dpi: int = 300,
    grid: bool = False,
    annotate_last: bool = True,
    bar_width: float = 0.08,
    capsize: int = 3,
    linewidth: float = 1.6,
    markersize: int = 5,
    cmap_name: str = "tab10",
):
    """
    One plot:
      x: p
      left y: retrieval (scatter curve with error bars)
      right y: centroid gap (bars with error bars)

    Two models (or more) shown in same plot.
    Each model's curve is aligned to its own bar centers (xs + offsets[mi]).
    Legend has one line per model: "Model: [curve] Retrieval, [bar] Gap"
    """
    models = sorted({r["Model"] for r in rows})
    ps = sorted({float(r["p"]) for r in rows})
    if len(models) == 0 or len(ps) == 0:
        print("[warn] empty rows")
        return

    color_map = _build_model_color_map(models, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax2 = ax.twinx()

    ax.set_xlabel("Corruption probability p")
    ax.set_ylabel(retrieval_label)
    ax2.set_ylabel("Centroid gap")

    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.35)
    else:
        ax.grid(False)

    for axis in (ax, ax2):
        for spine in axis.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
            spine.set_visible(True)

    ax.tick_params(direction="in", length=5, width=0.8, colors="black", top=True, right=False)
    ax2.tick_params(direction="in", length=5, width=0.8, colors="black", left=False, right=True)

    xs = np.array(ps, dtype=float)

    m = len(models)
    offsets = np.linspace(-(m - 1) / 2, (m - 1) / 2, m) * bar_width

    # Collect legend tuples (curve, bar) and labels (one line per model)
    legend_handles = []
    legend_labels = []

    for mi, model in enumerate(models):
        col = color_map[model]
        rr = [r for r in rows if r["Model"] == model]
        byp = {float(r["p"]): r for r in rr}

        retr = np.array([float(byp[p][retrieval_mean_key]) if p in byp else np.nan for p in ps], dtype=float)
        retr_std = np.array([float(byp[p][retrieval_std_key]) if p in byp else np.nan for p in ps], dtype=float)

        cen = np.array([float(byp[p]["Centroid_mean"]) if p in byp else np.nan for p in ps], dtype=float)
        cen_std = np.array([float(byp[p]["Centroid_std"]) if p in byp else np.nan for p in ps], dtype=float)

        # model-specific x positions (bar centers)
        x_model = xs + offsets[mi]

        # ---- Right axis: centroid bars ----
        ax2.bar(
            x_model,
            cen,
            width=bar_width * 0.92,
            color=col,
            alpha=0.25,
            edgecolor="black",
            linewidth=0.6,
            zorder=1,
        )
        ax2.errorbar(
            x_model,
            cen,
            yerr=cen_std,
            fmt="none",
            ecolor=col,
            elinewidth=1.2,
            capsize=capsize,
            zorder=2,
            alpha=0.9,
        )

        # ---- Left axis: retrieval curve aligned to bar centers ----
        ax.errorbar(
            x_model,
            retr,
            yerr=retr_std,
            fmt="o-",
            color=col,
            ecolor=col,
            linewidth=linewidth,
            markersize=markersize,
            capsize=capsize,
            zorder=4,
            alpha=0.95,
        )

        if annotate_last:
            idx_valid = np.where(~np.isnan(retr))[0]
            if len(idx_valid) > 0:
                j = int(idx_valid[-1])
                xj, yj = x_model[j], retr[j]
                dx = -10 if j == len(xs) - 1 else 10
                ha = "right" if dx < 0 else "left"
                ax.annotate(
                    model,
                    (xj, yj),
                    textcoords="offset points",
                    xytext=(dx, 10),
                    ha=ha,
                    va="bottom",
                    fontsize=14,
                    color=col,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.65),
                )

        # Legend proxies (same color)
        curve_proxy = plt.Line2D(
            [0], [0],
            color=col, marker="o", linestyle="-",
            linewidth=linewidth, markersize=markersize
        )
        bar_proxy = plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=col, alpha=0.25,
            edgecolor="black", linewidth=0.6
        )

        # One legend entry per model, text includes "Retrieval, Gap"
        legend_handles.append((curve_proxy, bar_proxy))
        legend_labels.append(f"{model}: Retrieval, Gap")

    # Legend: 2 lines (one per model), each line shows curve glyph + bar glyph
    from matplotlib.legend_handler import HandlerTuple
    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower left",
        frameon=True,
        edgecolor="gray",
        fancybox=False,
        framealpha=0.85,
        fontsize=10,
    )

    # x ticks at the true p values (center positions)
    # Note: curves/bars are offset around these tick marks.
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p:.2f}" for p in ps])

    plt.tight_layout()
    tag = retrieval_mean_key.lower().replace("_mean", "")
    out_png = f"{out_prefix}_p_curve_{tag}_bar_centroid.png"
    out_pdf = f"{out_prefix}_p_curve_{tag}_bar_centroid.pdf"
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="exp2")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--grid", action="store_true", help="Enable dashed/dotted grids (default off).")
    ap.add_argument("--which", choices=["avg", "i2t", "t2i", "all"], default="avg",
                    help="Which retrieval metric to plot on left axis.")
    args = ap.parse_args()

    raw = load_jsons(args.result_dir)
    if len(raw) == 0:
        raise RuntimeError(f"No exp2_..._p..._eval.json files found in: {args.result_dir}")

    groups = build_groups(raw)
    rows = aggregate_groups(groups)

    # Write table
    out_txt = f"{args.out_prefix}_table.txt"
    write_tables(rows, out_txt=out_txt, digits=args.digits)

    if args.which in ("avg", "all"):
        plot_exp2_curve_and_bar(
            rows,
            retrieval_mean_key="AVG_mean",
            retrieval_std_key="AVG_std",
            retrieval_label="Average retrieval R@1",
            out_prefix=args.out_prefix,
            dpi=args.dpi,
            grid=args.grid,
            annotate_last=True,
        )

    if args.which in ("i2t", "all"):
        plot_exp2_curve_and_bar(
            rows,
            retrieval_mean_key="I2T_mean",
            retrieval_std_key="I2T_std",
            retrieval_label="I→T retrieval R@1",
            out_prefix=args.out_prefix,
            dpi=args.dpi,
            grid=args.grid,
            annotate_last=True,
        )

    if args.which in ("t2i", "all"):
        plot_exp2_curve_and_bar(
            rows,
            retrieval_mean_key="T2I_mean",
            retrieval_std_key="T2I_std",
            retrieval_label="T→I retrieval R@1",
            out_prefix=args.out_prefix,
            dpi=args.dpi,
            grid=args.grid,
            annotate_last=True,
        )

if __name__ == "__main__":
    main()
    
    """
    python exp2_summarize_results.py --result_dir exp2_results --out_prefix exp2_summary/exp2 --which all --annotate --dpi 600 --grid
    """