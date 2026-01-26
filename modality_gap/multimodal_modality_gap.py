import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------
# Plot style
# ----------------------------
def set_plot_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
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


def wrap_pi(theta: torch.Tensor):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def wrap_pi_np(x: np.ndarray):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ----------------------------
# Latent sampler (non-uniform angles): mixture of von Mises
# ----------------------------
def sample_theta_mixture(n, w=0.7, mu1=0.0, mu2=math.pi, kappa=6.0, device="cpu"):
    comp = torch.bernoulli(torch.full((n,), w, device=device)).bool()
    vm = torch.distributions.VonMises
    t1 = vm(torch.tensor(mu1, device=device), torch.tensor(kappa, device=device)).sample((n,))
    t2 = vm(torch.tensor(mu2, device=device), torch.tensor(kappa, device=device)).sample((n,))
    theta = torch.where(comp, t1, t2)
    return wrap_pi(theta)


def features_from_theta(theta, obs_noise=0.02):
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    if obs_noise > 0:
        x = x + obs_noise * torch.randn_like(x)
    return x


# ----------------------------
# Encoders
# ----------------------------
class LinearEncoder(torch.nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.W = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.W(x)


def angle_of(z: torch.Tensor):
    return torch.atan2(z[:, 1], z[:, 0])


# ----------------------------
# Cosine / sphere critic + symmetric CLIP loss
# ----------------------------
def sim_matrix_cosine(z1: torch.Tensor, z2: torch.Tensor):
    # z1:[B,d], z2:[B,d] -> [B,B]
    return z1 @ z2.t()


def symmetric_clip_loss_cosine(z1: torch.Tensor, z2: torch.Tensor, tau: float):
    S = sim_matrix_cosine(z1, z2) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(S, labels)
    loss21 = F.cross_entropy(S.t(), labels)
    return 0.5 * (loss12 + loss21)


# ----------------------------
# Gap metric: symmetric KL between angle histograms
# ----------------------------
def sym_kl_from_angles_np(a1: np.ndarray, a2: np.ndarray, nbins=60, eps=1e-8):
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    h1, _ = np.histogram(a1, bins=bins, density=False)
    h2, _ = np.histogram(a2, bins=bins, density=False)

    p = (h1.astype(np.float64) + eps)
    q = (h2.astype(np.float64) + eps)
    p = p / p.sum()
    q = q / q.sum()

    dkl_pq = np.sum(p * (np.log(p) - np.log(q)))
    dkl_qp = np.sum(q * (np.log(q) - np.log(p)))
    return float(dkl_pq + dkl_qp)


# ----------------------------
# Train + evaluate
# ----------------------------
def run_multimodal_cosine(seed=0, device="cpu",
                         steps=2000, B=256, lr=5e-3,
                         tau=0.07,
                         misalign_sigma=0.0,
                         obs_noise=0.02,
                         mix_w=0.7, kappa=6.0,
                         n_eval=8000,
                         return_repr=False):
    set_seed(seed)

    f = LinearEncoder(2, 2).to(device)
    g = LinearEncoder(2, 2).to(device)
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=lr)

    # training (feature-normalized)
    for _ in range(steps):
        theta = sample_theta_mixture(B, w=mix_w, kappa=kappa, device=device)
        theta2 = wrap_pi(theta + misalign_sigma * torch.randn(B, device=device))

        x1 = features_from_theta(theta, obs_noise=obs_noise).to(device)
        x2 = features_from_theta(theta2, obs_noise=obs_noise).to(device)

        z1 = F.normalize(f(x1), dim=1)
        z2 = F.normalize(g(x2), dim=1)

        loss = symmetric_clip_loss_cosine(z1, z2, tau=tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # evaluation
    theta = sample_theta_mixture(n_eval, w=mix_w, kappa=kappa, device=device)
    theta2 = wrap_pi(theta + misalign_sigma * torch.randn(n_eval, device=device))

    x1 = features_from_theta(theta, obs_noise=obs_noise).to(device)
    x2 = features_from_theta(theta2, obs_noise=obs_noise).to(device)

    with torch.no_grad():
        z1u = F.normalize(f(x1), dim=1)
        z2u = F.normalize(g(x2), dim=1)
        a1 = angle_of(z1u)
        a2 = angle_of(z2u)

    a1_np = a1.detach().cpu().numpy()
    a2_np = a2.detach().cpu().numpy()
    gap = sym_kl_from_angles_np(a1_np, a2_np, nbins=60)

    if return_repr:
        return gap, z1u.detach().cpu().numpy(), z2u.detach().cpu().numpy(), a1_np, a2_np
    return gap


# ----------------------------
# Plots
# ----------------------------
def plot_gap_curve(sigmas, gap_means, gap_stds, n_seeds, title, outpath=None):
    plt.figure(figsize=(3.35, 2.2))
    
    gap_sems = gap_stds / np.sqrt(n_seeds)
    plt.fill_between(sigmas, gap_means - gap_sems, gap_means + gap_sems, color='C0', alpha=0.2, linewidth=0, zorder=0)
    plt.plot(sigmas, gap_means, marker="o", markersize=4, color='C0', linewidth=1.1, zorder=1, label="Mean $\pm$ SEM")

    plt.xlabel(r"Misalignment scale $\sigma_{\mathrm{mis}}$")
    plt.ylabel(r"$D_{\mathrm{KL}}^{\mathrm{sym}}(\hat q_\theta,\hat q_\phi)$")
    plt.title(title, fontsize=8, pad=2)

    ax = plt.gca()
    prettify_ax(ax)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=600)
        plt.close()
    else:
        plt.show()


def plot_polar_density(a1, a2, title, outpath=None, nbins=40):
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    c = 0.5 * (bins[:-1] + bins[1:])

    h1, _ = np.histogram(a1, bins=bins, density=True)
    h2, _ = np.histogram(a2, bins=bins, density=True)

    # close curves
    c2 = np.concatenate([c, c[:1]])
    h1c = np.concatenate([h1, h1[:1]])
    h2c = np.concatenate([h2, h2[:1]])

    plt.figure(figsize=(3.35, 3.0))
    ax = plt.subplot(111, projection="polar")
    ax.plot(c2, h1c, linewidth=1.2, label="Modality 1")
    ax.plot(c2, h2c, linewidth=1.2, label="Modality 2")
    ax.set_title(title, fontsize=8, pad=10)
    ax.set_rticks([])

    leg = ax.legend(loc="best", frameon=True, fancybox=True,
                    framealpha=0.90, borderpad=0.35, handletextpad=0.4)
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=600)
        plt.close()
    else:
        plt.show()


def plot_joint_angle_heatmap(a1, a2, title, outpath=None, nbins=80):
    plt.figure(figsize=(3.35, 3.0))
    plt.hist2d(a1, a2, bins=nbins, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    plt.plot([-np.pi, np.pi], [-np.pi, np.pi], linestyle="--", linewidth=0.9, color="0.35")
    plt.xlabel(r"Angle $a_1$ (modality 1)")
    plt.ylabel(r"Angle $a_2$ (modality 2)")
    plt.title(title, fontsize=8, pad=2)
    plt.colorbar(label="Count")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=600)
        plt.close()
    else:
        plt.show()


def plot_delta_density(a1, a2, title, outpath=None, nbins=60):
    delta = wrap_pi_np(a2 - a1)
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    h, _ = np.histogram(delta, bins=bins, density=True)

    plt.figure(figsize=(3.35, 2.2))
    plt.plot(centers, h, linewidth=1.2)
    plt.xlabel(r"$\Delta a = \mathrm{wrap}(a_2-a_1)$")
    plt.ylabel("Density")
    plt.title(title, fontsize=8, pad=2)

    ax = plt.gca()
    prettify_ax(ax)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=600)
        plt.close()
    else:
        plt.show()


def main():
    set_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    NSeeds = 20
    seeds = list(range(NSeeds))
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Hyperparams
    steps = 2000
    B = 256
    lr = 5e-3
    tau = 0.07
    obs_noise = 0.02
    n_eval = 8000
    mix_w = 0.7
    kappa = 6.0

    gap_means, gap_stds = [], []
    repr_cache = {}  # store one representative seed's repr per sigma

    for sig in sigmas:
        gaps = []
        for s in seeds:
            if s == seeds[0]:
                gap, z1u, z2u, a1, a2 = run_multimodal_cosine(
                    seed=s, device=device,
                    steps=steps, B=B, lr=lr, tau=tau,
                    misalign_sigma=sig,
                    obs_noise=obs_noise,
                    mix_w=mix_w, kappa=kappa,
                    n_eval=n_eval,
                    return_repr=True
                )
                repr_cache[sig] = (a1, a2)  # only angles needed for visuals
            else:
                gap = run_multimodal_cosine(
                    seed=s, device=device,
                    steps=steps, B=B, lr=lr, tau=tau,
                    misalign_sigma=sig,
                    obs_noise=obs_noise,
                    mix_w=mix_w, kappa=kappa,
                    n_eval=n_eval,
                    return_repr=False
                )
            gaps.append(gap)

        gaps = np.array(gaps)
        gap_means.append(gaps.mean())
        gap_stds.append(gaps.std())
        print(f"[cosine] sigma={sig:.2f}: symKL mean±std = {gaps.mean():.4f} ± {gaps.std():.4f}")

    # curve
    plot_gap_curve(
        sigmas, gap_means, gap_stds, n_seeds = NSeeds,
        title="Multimodal marginal gap vs misalignment",
        outpath="mm_gapcurve_cosine.pdf"
    )

    # per-sigma visuals (from seed 0)
    for sig in sigmas:
        a1, a2 = repr_cache[sig]

        plot_polar_density(
            a1, a2,
            title=f"Cosine: angle density ($\\sigma_{{mis}}={sig:.2f}$)",
            outpath=f"mm_polar_cosine_sig{sig:.2f}.pdf"
        )

        plot_joint_angle_heatmap(
            a1, a2,
            title=f"Cosine: joint angles ($\\sigma_{{mis}}={sig:.2f}$)",
            outpath=f"mm_joint_cosine_sig{sig:.2f}.pdf"
        )

        plot_delta_density(
            a1, a2,
            title=f"Cosine: angle shift $\\Delta a$ ($\\sigma_{{mis}}={sig:.2f}$)",
            outpath=f"mm_delta_cosine_sig{sig:.2f}.pdf"
        )

    print("Saved: mm_gapcurve_cosine.pdf, mm_polar_*.pdf, mm_joint_*.pdf, mm_delta_*.pdf")


if __name__ == "__main__":
    main()
