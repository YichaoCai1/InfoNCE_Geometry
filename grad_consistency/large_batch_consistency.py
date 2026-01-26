import math
import numpy as np
import torch
import matplotlib.ticker as mticker
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------
# Plot style
# ----------------------------
def set_plot_style():
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # lighter strokes
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "savefig.dpi": 600,
    })


def prettify_ax(ax):
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
# ℓ = -(1/τ)s_pos + log( exp(s_pos/τ) + Σ exp(s_neg/τ) )
# Here we average over B anchors for stability.
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
        # dist^2(z,w) = ||z||^2 + ||w||^2 - 2 z·w
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
    x = sample_mog(B, m, k=mog_k, sep=mog_sep, sigma=mog_sigma, device=device)      # [B,m]
    x_pos = x + aug_use * torch.randn_like(x)                                       # [B,m]
    x_negs = sample_mog(Nref, m, k=mog_k, sep=mog_sep, sigma=mog_sigma, device=device)  # [Nref,m]

    z = encode(W, x, mode=emb_mode)            # [B,d]
    v = encode(W, x_pos, mode=emb_mode)        # [B,d]
    w_all = encode(W, x_negs, mode=emb_mode)   # [Nref,d]

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


def aggregate_and_save(Ns, all_cos, all_err, n_seeds, tag: str):
    cos_mean, cos_std = all_cos.mean(0), all_cos.std(0)
    err_mean, err_std = all_err.mean(0), all_err.std(0)
    
    # standard errors
    cos_sem = cos_std / np.sqrt(n_seeds)
    err_sem = err_std / np.sqrt(n_seeds)
    
    # Plot 1: cosine similarity
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.fill_between(Ns, cos_mean - cos_sem, cos_mean + cos_sem, color='C0', alpha=0.2, linewidth=0, zorder=0)
    ax.plot(Ns, cos_mean, marker="o", markersize=4, color='C0', linewidth=1.1, zorder=1, label="Mean $\pm$ SEM")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_ylim(0.2, 1.01)
    ax.set_xlabel("Number of negatives $N$")
    ax.set_ylabel(r"$\cos(\mathsf{g}_N, \mathsf{g}_{\mathrm{ref}})$")
    ax.set_title(f"Gradient alignment vs $N$")
    prettify_ax(ax)
    fig.tight_layout()
    fig.savefig(f"grad_alignment_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: relative error
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.fill_between(Ns, err_mean - err_sem, err_mean + err_sem, color='C0', alpha=0.2, linewidth=0, zorder=0)
    ax.plot(Ns, err_mean, marker="o", markersize=4, color='C0', linewidth=1.1, zorder=1, label="Mean $\pm$ SEM")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of negatives $N$")
    ax.set_ylabel(r"$\|\mathsf{g}_N-\mathsf{g}_{\mathrm{ref}}\|/\|\mathsf{g}_{\mathrm{ref}}\|$")
    ax.set_title(f"Relative gradient error vs $N$")
    prettify_ax(ax)
    fig.tight_layout()
    fig.savefig(f"grad_relerr_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    set_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    Ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    Nref = 4096
    Nseeds = 20
    seeds = range(Nseeds)

    for kind in ["cosine", "rbf"]:
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

        tag = "cosine_sphere" if kind == "cosine" else "rbf_bounded"
        aggregate_and_save(Ns, all_cos, all_err, n_seeds=Nseeds, tag=tag)

    print("Saved: grad_alignment_*.pdf and grad_relerr_*.pdf")


if __name__ == "__main__":
    main()
