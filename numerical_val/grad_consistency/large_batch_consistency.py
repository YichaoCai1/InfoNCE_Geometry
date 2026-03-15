import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

NUMERICAL_ROOT = Path(__file__).resolve().parents[1]
if str(NUMERICAL_ROOT) not in sys.path:
    sys.path.append(str(NUMERICAL_ROOT))

from plot_style import (
    CURVE_FIGSIZE,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    prettify_ax,
    set_plot_style,
    styled_legend,
)

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ----------------------------
# Data: mixture of Gaussians
# ----------------------------
def sample_mog(n: int, m: int, k: int = 4, sep: float = 4.0, sigma: float = 1.0, device="cpu"):
    means = torch.randn(k, m, device=device)
    means = means / (means.norm(dim=1, keepdim=True) + 1e-12)
    means = means * sep
    comp = torch.randint(low=0, high=k, size=(n,), device=device)
    return means[comp] + sigma * torch.randn(n, m, device=device)

# ----------------------------
# Encoder
# ----------------------------
def encode(W: torch.Tensor, x: torch.Tensor, mode: str):
    z = x @ W.t()
    if mode == "sphere":
        z = F.normalize(z, dim=-1)
    elif mode == "euclid":
        z = torch.tanh(z)  # bounded embeddings
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return z

# ----------------------------
# Vectorized loss matching your ℓ_B form
# ----------------------------
def unimodal_infonce_loss_batch(z, v, wnegs, tau: float, kind: str, rbf_scale: float = 1.0):
    """
    z, v: [B, d]
    wnegs: [N, d]
    """
    B, d = z.shape
    N = wnegs.shape[0]

    if kind == "cosine":
        # s = <z,w>
        s_pos = (z * v).sum(dim=1)               # [B]
        s_negs = (z @ wnegs.t())                 # [B, N]
    elif kind == "rbf":
        # s = -scale * ||z-w||^2
        z2 = (z * z).sum(dim=1, keepdim=True)    # [B,1]
        w2 = (wnegs * wnegs).sum(dim=1).view(1, N)  # [1,N]
        zv = z @ wnegs.t()                       # [B,N]
        dist2_negs = z2 + w2 - 2.0 * zv          # [B,N]

        dist2_pos = ((z - v) ** 2).sum(dim=1)    # [B]
        s_pos = -rbf_scale * dist2_pos
        s_negs = -rbf_scale * dist2_negs
    else:
        raise ValueError(f"Unknown critic kind: {kind}")

    logits = torch.cat([s_pos.view(B, 1), s_negs], dim=1) / tau   # [B, 1+N]
    logZ = torch.logsumexp(logits, dim=1)                         # [B]
    loss = -(s_pos / tau) + logZ                                  # [B]
    return loss.mean()

def flat_grad(g: torch.Tensor):
    return g.detach().reshape(-1)

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    return F.cosine_similarity(a, b, dim=0).item()

def rel_err(a: torch.Tensor, b: torch.Tensor, eps=1e-12):
    return (a - b).norm().item() / (b.norm().item() + eps)

# ----------------------------
# One-seed run
# ----------------------------
def run_one_seed(seed: int, Ns, Nref: int, B: int = 64,
                 m=64, d=128, tau=0.1, aug_sigma=0.05,
                 mog_k=4, mog_sep=4.0, mog_sigma=1.0,
                 critic_kind="cosine", device="cpu"):

    set_seed(seed)

    # critic-specific stabilization
    if critic_kind == "cosine":
        emb_mode = "sphere"
        tau_use = tau
        aug_use = aug_sigma
        rbf_scale = 1.0
    elif critic_kind == "rbf":
        emb_mode = "euclid"
        rbf_scale = 1.0 / d
        tau_use = 1.0
        aug_use = max(aug_sigma, 0.2)
    else:
        raise ValueError(critic_kind)

    W = (torch.randn(d, m, device=device) / math.sqrt(m)).requires_grad_(True)

    # B anchors/positives + shared negative pool
    x = sample_mog(B, m, k=mog_k, sep=mog_sep, sigma=mog_sigma, device=device)      
    x_pos = x + aug_use * torch.randn_like(x)                                       
    x_negs = sample_mog(Nref, m, k=mog_k, sep=mog_sep, sigma=mog_sigma, device=device)  

    z = encode(W, x, mode=emb_mode)            
    v = encode(W, x_pos, mode=emb_mode)        
    w_all = encode(W, x_negs, mode=emb_mode)   

    # reference gradient
    loss_ref = unimodal_infonce_loss_batch(z, v, w_all, tau=tau_use, kind=critic_kind, rbf_scale=rbf_scale)
    g_ref = torch.autograd.grad(loss_ref, W, retain_graph=True)[0]
    g_ref = flat_grad(g_ref).clone()

    cos_vals, err_vals = [], []
    for N in Ns:
        loss_N = unimodal_infonce_loss_batch(z, v, w_all[:N], tau=tau_use, kind=critic_kind, rbf_scale=rbf_scale)
        gN = torch.autograd.grad(loss_N, W, retain_graph=True)[0]
        gN = flat_grad(gN).clone()
        cos_vals.append(cos_sim(gN, g_ref))
        err_vals.append(rel_err(gN, g_ref))

    return np.array(cos_vals), np.array(err_vals)

# ----------------------------
# Plotting Comparison (Two distinct figures)
# ----------------------------
def plot_method_comparison(Ns, results_dict):
    """
    Plots the alignment and relative error for multiple methods on separate figures.
    results_dict: dict of the form { "Method Name": {"cos": all_cos_array, "err": all_err_array} }
    """
    # Create two separate figures
    fig1, ax1 = plt.subplots(figsize=CURVE_FIGSIZE)
    fig2, ax2 = plt.subplots(figsize=CURVE_FIGSIZE)

    colors = [PRIMARY_COLOR, SECONDARY_COLOR]
    
    for idx, (label_tag, data) in enumerate(results_dict.items()):
        all_cos = data["cos"]
        all_err = data["err"]
        
        cos_mean, cos_std = all_cos.mean(0), all_cos.std(0)
        err_mean, err_std = all_err.mean(0), all_err.std(0)
        
        c = colors[idx % len(colors)]
        
        # --- Figure 1: Cosine similarity ---
        ax1.fill_between(Ns, cos_mean - cos_std, cos_mean + cos_std, color=c, alpha=0.18, linewidth=0, zorder=0)
        ax1.plot(Ns, cos_mean, marker="o", color=c, zorder=1, label=label_tag)

        # --- Figure 2: Relative error ---
        ax2.fill_between(Ns, err_mean - err_std, err_mean + err_std, color=c, alpha=0.18, linewidth=0, zorder=0)
        ax2.plot(Ns, err_mean, marker="o", color=c, zorder=1, label=label_tag)

    # Format Figure 1 (Alignment)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(Ns)
    ax1.set_xticklabels([str(n) for n in Ns])
    ax1.set_ylim(0.2, 1.05)
    ax1.set_xlabel("Number of negatives $N$")
    ax1.set_ylabel(r"$\cos(\mathsf{g}_N, \mathsf{g}_{\mathrm{ref}})$")
    ax1.set_title("Gradient alignment vs $N$")
    styled_legend(ax1, loc="lower right")
    prettify_ax(ax1)
    
    fig1.tight_layout()
    fig1.savefig("grad_alignment_comparison.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Format Figure 2 (Relative Error)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(Ns)
    ax2.set_xticklabels([str(n) for n in Ns])
    ax2.set_xlabel("Number of negatives $N$")
    ax2.set_ylabel(r"$\|\mathsf{g}_N-\mathsf{g}_{\mathrm{ref}}\|/\|\mathsf{g}_{\mathrm{ref}}\|$")
    ax2.set_title("Relative gradient error vs $N$")
    styled_legend(ax2, loc="upper right")
    prettify_ax(ax2)
    
    fig2.tight_layout()
    fig2.savefig("grad_relerr_comparison.pdf", bbox_inches="tight")
    plt.close(fig2)

# ----------------------------
# Main
# ----------------------------
def main():
    set_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    Ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    Nref = 4096
    Nseeds = 20
    seeds = range(Nseeds)

    # Dictionary to store results for both methods
    results = {}

    for kind in ["cosine", "rbf"]:
        print(f"Running simulation for: {kind}...")
        cos_list, err_list = [], []
        for s in seeds:
            c, e = run_one_seed(
                seed=s, Ns=Ns, Nref=Nref, B=64,
                m=64, d=128, tau=0.1, aug_sigma=0.05,
                mog_k=4, mog_sep=4.0, mog_sigma=1.0,
                critic_kind=kind, device=device
            )
            cos_list.append(c)
            err_list.append(e)

        all_cos = np.stack(cos_list, axis=0)
        all_err = np.stack(err_list, axis=0)
        
        # Create a clean label for the legend
        label = "Cosine (Sphere)" if kind == "cosine" else "RBF (Bounded)"
        results[label] = {"cos": all_cos, "err": all_err}

    # Plot everything together into two files
    print("Generating plots...")
    plot_method_comparison(Ns, results)
    print("Saved: grad_alignment_comparison.pdf and grad_relerr_comparison.pdf")

if __name__ == "__main__":
    main()
