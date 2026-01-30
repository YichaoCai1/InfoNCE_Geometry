import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------------------
# Plot style
# ----------------------------
def set_plot_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 300,
    })


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def wrap_pi(theta: torch.Tensor):
    return (theta + math.pi) % (2 * math.pi) - math.pi


# ----------------------------
# Latent sampler (mixture of von Mises)
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
# Symmetric CLIP loss (cosine)
# ----------------------------
def sim_matrix_cosine(z1: torch.Tensor, z2: torch.Tensor):
    return z1 @ z2.t()


def symmetric_clip_loss_cosine(z1: torch.Tensor, z2: torch.Tensor, tau: float):
    S = sim_matrix_cosine(z1, z2) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(S, labels)
    loss21 = F.cross_entropy(S.t(), labels)
    return 0.5 * (loss12 + loss21)


# ----------------------------
# Joint histogram (log-compressed for visibility)
# ----------------------------
def joint_hist2d_logcounts(a1_np: np.ndarray, a2_np: np.ndarray, nbins=80):
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    H, _, _ = np.histogram2d(
        a1_np, a2_np,
        bins=[bins, bins],
        range=[[-np.pi, np.pi], [-np.pi, np.pi]]
    )
    H = H.astype(np.float32)
    H = np.log1p(H)  # CRITICAL: reveals structure
    return H


@torch.no_grad()
def eval_angles(f, g, theta_eval, theta2_eval, obs_noise=0.02):
    x1 = features_from_theta(theta_eval, obs_noise=obs_noise)
    x2 = features_from_theta(theta2_eval, obs_noise=obs_noise)
    z1 = F.normalize(f(x1), dim=1)
    z2 = F.normalize(g(x2), dim=1)
    a1 = angle_of(z1).detach().cpu().numpy()
    a2 = angle_of(z2).detach().cpu().numpy()
    return a1, a2


def run_multimodal_cosine_with_joint_anim_data(
    seed=0,
    device="cpu",
    steps=2000,
    B=256,
    lr=5e-3,
    tau=0.07,
    misalign_sigma=0.0,
    obs_noise=0.02,
    mix_w=0.7,
    kappa=6.0,
    n_eval=8000,
    nbins=80,
    log_every=10,
    theta_eval=None,  # pass shared theta_eval across sigmas if you want
):
    """
    Experimental setting unchanged; logs joint-angle histograms:
    1 frame per log_every steps + initial frame at step 0.
    """
    set_seed(seed)

    f = LinearEncoder(2, 2).to(device)
    g = LinearEncoder(2, 2).to(device)
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=lr)

    if theta_eval is None:
        theta_eval = sample_theta_mixture(n_eval, w=mix_w, kappa=kappa, device=device)
    else:
        theta_eval = theta_eval.to(device)

    theta2_eval = wrap_pi(theta_eval + misalign_sigma * torch.randn_like(theta_eval))

    frames = []
    frame_steps = []

    # frame 0
    a1_0, a2_0 = eval_angles(f, g, theta_eval, theta2_eval, obs_noise=obs_noise)
    frames.append(joint_hist2d_logcounts(a1_0, a2_0, nbins=nbins))
    frame_steps.append(0)

    for t in range(steps):
        theta = sample_theta_mixture(B, w=mix_w, kappa=kappa, device=device)
        theta2 = wrap_pi(theta + misalign_sigma * torch.randn(B, device=device))

        x1 = features_from_theta(theta, obs_noise=obs_noise)
        x2 = features_from_theta(theta2, obs_noise=obs_noise)

        z1 = F.normalize(f(x1), dim=1)
        z2 = F.normalize(g(x2), dim=1)

        loss = symmetric_clip_loss_cosine(z1, z2, tau=tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step_now = t + 1
        if step_now % log_every == 0 or step_now == steps:
            a1, a2 = eval_angles(f, g, theta_eval, theta2_eval, obs_noise=obs_noise)
            frames.append(joint_hist2d_logcounts(a1, a2, nbins=nbins))
            frame_steps.append(step_now)

    return frames, frame_steps


# ----------------------------
# Animation: 2x4 panels, EACH panel has its OWN colorbar
# - colorbar scales can update per frame (dynamic min/max per panel)
# - each panel is square (1:1 width:height)
# ----------------------------
def animate_joint_angle_heatmap_grid_individual_cbars(
    frames_by_sigma,          # dict: sigma -> list of H (len = T)
    frame_steps,              # list of ints (len = T)
    sigmas,                   # list length 8
    outpath="mm_joint_grid_individual_cbars.gif",
    fps=12,
    q_vmax=0.995,             # use quantile vmax to avoid single-bin spikes
    cmap="magma",
):
    set_plot_style()

    nrows, ncols = 2, 4
    assert len(sigmas) == nrows * ncols, "sigmas must have length 8 for a 2x4 grid."

    T = len(frame_steps)
    for sig in sigmas:
        assert len(frames_by_sigma[sig]) == T, "All sigmas must have the same number of frames."

    # Figure size: choose per-panel size; aspect set to equal makes each axis square in data coords.
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 3.20 * nrows),
        squeeze=False
    )

    extent = [-np.pi, np.pi, -np.pi, np.pi]

    ims = []
    cbars = []

    # Build each panel + its own colorbar axis
    for idx, sig in enumerate(sigmas):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        H0 = frames_by_sigma[sig][0]

        # initial scaling for this panel
        vmax0 = float(np.quantile(H0, q_vmax))
        vmax0 = max(vmax0, 1e-6)

        im = ax.imshow(
            H0.T,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            aspect="equal",               # 1:1 ratio for the heatmap itself
            vmin=0.0,
            vmax=vmax0,
            cmap=cmap,
        )

        # Force the axes box to be square (panel 1:1 in display too)
        ax.set_aspect("equal", adjustable="box")

        # Diagonal guide
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], linestyle="--", linewidth=0.9, color="0.35")

        ax.set_title(rf"$\sigma_{{mis}}={sig:.2f}$", fontsize=10, pad=2)

        # Light labeling (avoid clutter)
        if c == 0:
            ax.set_ylabel(r"$a_2$")
        else:
            ax.set_yticklabels([])

        if r == nrows - 1:
            ax.set_xlabel(r"$a_1$")
        else:
            ax.set_xticklabels([])

        # attach individual colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.06)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("log(1+count)", fontsize=8, labelpad=4)

        ims.append(im)
        cbars.append(cbar)

    # global title updated per frame
    supt = fig.suptitle(f"Cosine: joint angles, step={frame_steps[0]}", fontsize=12, y=0.98)

    # Manual spacing (more reliable than tight_layout with many cbars)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.10, top=0.90, wspace=0.25, hspace=0.30)

    def _update(t):
        supt.set_text(f"Cosine: joint angles, step={frame_steps[t]}")
        for idx, sig in enumerate(sigmas):
            H = frames_by_sigma[sig][t]
            ims[idx].set_data(H.T)

            # dynamic scaling PER PANEL PER FRAME
            vmax = float(np.quantile(H, q_vmax))
            vmax = max(vmax, 1e-6)
            ims[idx].set_clim(0.0, vmax)
            cbars[idx].update_normal(ims[idx])

        return ims + [supt]

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=int(1000 / fps), blit=False
    )

    if outpath.lower().endswith(".gif"):
        anim.save(outpath, writer=animation.PillowWriter(fps=fps))
    elif outpath.lower().endswith(".mp4"):
        anim.save(outpath, writer=animation.FFMpegWriter(fps=fps))
    else:
        raise ValueError("outpath must end with .gif or .mp4")

    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    set_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # Experimental setting (UNCHANGED)
    seed = 0
    steps = 2000
    B = 256
    lr = 5e-3
    tau = 0.07
    obs_noise = 0.02
    n_eval = 8000
    mix_w = 0.7
    kappa = 6.0

    # Animation settings
    log_every = 10
    nbins = 80
    fps = 12

    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Share theta_eval across all sigmas for comparable panels
    set_seed(seed)
    theta_eval_shared = sample_theta_mixture(n_eval, w=mix_w, kappa=kappa, device=device)

    frames_by_sigma = {}
    frame_steps_ref = None

    for sig in sigmas:
        print(f"Running sigma={sig:.2f} ...")
        frames, frame_steps = run_multimodal_cosine_with_joint_anim_data(
            seed=seed,
            device=device,
            steps=steps,
            B=B,
            lr=lr,
            tau=tau,
            misalign_sigma=sig,
            obs_noise=obs_noise,
            mix_w=mix_w,
            kappa=kappa,
            n_eval=n_eval,
            nbins=nbins,
            log_every=log_every,
            theta_eval=theta_eval_shared,
        )
        frames_by_sigma[sig] = frames
        if frame_steps_ref is None:
            frame_steps_ref = frame_steps
        else:
            if frame_steps_ref != frame_steps:
                raise RuntimeError("Frame steps mismatch across sigmas.")

    outpath = "mm_joint_cosine_anim_grid_2x4_individual_cbars.gif"
    animate_joint_angle_heatmap_grid_individual_cbars(
        frames_by_sigma=frames_by_sigma,
        frame_steps=frame_steps_ref,
        sigmas=sigmas,
        outpath=outpath,
        fps=fps,
        q_vmax=0.995,   # robust dynamic scaling per panel per frame
        cmap="magma",
    )
    print("Saved:", outpath)


if __name__ == "__main__":
    main()
