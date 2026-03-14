import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


NUMERICAL_ROOT = Path(__file__).resolve().parents[1]
if str(NUMERICAL_ROOT) not in sys.path:
    sys.path.append(str(NUMERICAL_ROOT))

from plot_style import (
    CURVE_FIGSIZE,
    DIAGONAL_COLOR,
    GRID_COLOR,
    HEATMAP_CMAP,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SQUARE_FIGSIZE,
    prettify_ax,
    set_plot_style,
    styled_legend,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def wrap_pi(theta: torch.Tensor):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def wrap_pi_np(x: np.ndarray):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _uses_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


def _capture_rng_state(device):
    state = {"cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
    if _uses_cuda(device):
        cuda_device = torch.device(device)
        cuda_index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
        state["cuda_device"] = cuda_index
        state["cuda"] = torch.cuda.get_rng_state(cuda_index)
    return state


def _restore_rng_state(state):
    torch.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], state["cuda_device"])


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


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.W = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.W(x)


def build_linear_encoders(device="cpu"):
    f = LinearEncoder(2, 2).to(device)
    g = LinearEncoder(2, 2).to(device)
    return f, g


def angle_of(z: torch.Tensor):
    return torch.atan2(z[:, 1], z[:, 0])


def sim_matrix_cosine(z1: torch.Tensor, z2: torch.Tensor):
    return z1 @ z2.t()


def symmetric_clip_loss_cosine(z1: torch.Tensor, z2: torch.Tensor, tau: float):
    logits = sim_matrix_cosine(z1, z2) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(logits, labels)
    loss21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss12 + loss21)


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


def summarize_seed_gaps(gaps, seeds):
    gaps = np.asarray(gaps, dtype=float)
    if gaps.ndim != 1 or gaps.size == 0:
        raise ValueError("Expected a non-empty 1D array of seed gaps.")
    center = float(gaps.mean())
    std = float(gaps.std(ddof=0))
    band_low = center - std
    band_high = center + std
    label = "Mean +/- std"
    short_label = "mean +/- std"
    rep_idx = int(np.argmin(np.abs(gaps - center)))

    return {
        "center": center,
        "band_low": band_low,
        "band_high": band_high,
        "label": label,
        "short_label": short_label,
        "rep_seed": seeds[rep_idx],
        "raw_mean": float(gaps.mean()),
        "raw_std": float(gaps.std(ddof=0)),
        "n_total": int(len(gaps)),
    }


def joint_hist2d_logcounts(a1_np: np.ndarray, a2_np: np.ndarray, nbins=80):
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    hist, _, _ = np.histogram2d(
        a1_np,
        a2_np,
        bins=[bins, bins],
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
    )
    return np.log1p(hist.astype(np.float32))


@torch.no_grad()
def eval_angles(f, g, theta_eval, theta2_eval, obs_noise=0.02):
    x1 = features_from_theta(theta_eval, obs_noise=obs_noise)
    x2 = features_from_theta(theta2_eval, obs_noise=obs_noise)
    z1 = F.normalize(f(x1), dim=1)
    z2 = F.normalize(g(x2), dim=1)
    a1 = angle_of(z1).detach().cpu().numpy()
    a2 = angle_of(z2).detach().cpu().numpy()
    return a1, a2


def run_multimodal_cosine(
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
    return_repr=False,
    return_hist=False,
    nbins=80,
    log_every=10,
    theta_eval=None,
):
    set_seed(seed)

    f, g = build_linear_encoders(device=device)
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=lr)

    if theta_eval is not None:
        theta_eval = theta_eval.to(device)
    theta2_eval = None

    frames = []
    frame_steps = []
    if return_hist:
        rng_state = _capture_rng_state(device)
        if theta_eval is None:
            theta_eval = sample_theta_mixture(n_eval, w=mix_w, kappa=kappa, device=device)
        theta2_eval = wrap_pi(theta_eval + misalign_sigma * torch.randn_like(theta_eval))
        a1_0, a2_0 = eval_angles(f, g, theta_eval, theta2_eval, obs_noise=obs_noise)
        frames.append(joint_hist2d_logcounts(a1_0, a2_0, nbins=nbins))
        frame_steps.append(0)
        _restore_rng_state(rng_state)

    for step in range(steps):
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

        step_now = step + 1
        if return_hist and (step_now % log_every == 0 or step_now == steps):
            a1_hist, a2_hist = eval_angles(f, g, theta_eval, theta2_eval, obs_noise=obs_noise)
            frames.append(joint_hist2d_logcounts(a1_hist, a2_hist, nbins=nbins))
            frame_steps.append(step_now)

    if theta_eval is None:
        theta_eval = sample_theta_mixture(n_eval, w=mix_w, kappa=kappa, device=device)
    if theta2_eval is None:
        theta2_eval = wrap_pi(theta_eval + misalign_sigma * torch.randn_like(theta_eval))

    with torch.no_grad():
        x1 = features_from_theta(theta_eval, obs_noise=obs_noise).to(device)
        x2 = features_from_theta(theta2_eval, obs_noise=obs_noise).to(device)
        z1u = F.normalize(f(x1), dim=1)
        z2u = F.normalize(g(x2), dim=1)
        a1 = angle_of(z1u)
        a2 = angle_of(z2u)

    a1_np = a1.detach().cpu().numpy()
    a2_np = a2.detach().cpu().numpy()

    result = {"gap": sym_kl_from_angles_np(a1_np, a2_np, nbins=60)}
    if return_repr:
        result["angles"] = (a1_np, a2_np)
    if return_hist:
        result["frames"] = frames
        result["frame_steps"] = frame_steps
    return result


def plot_gap_curve(sigmas, centers, band_lows, band_highs, title, outpath, summary_label):
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    ax.fill_between(
        sigmas,
        band_lows,
        band_highs,
        color=PRIMARY_COLOR,
        alpha=0.18,
        linewidth=0,
        zorder=0,
    )
    ax.plot(sigmas, centers, marker="o", color=PRIMARY_COLOR, zorder=1, label=summary_label)
    ax.set_xlabel(r"Misalignment scale $\sigma_{\mathrm{mis}}$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{sym}}(\hat q_\theta,\hat q_\phi)$")
    ax.set_title(title)
    prettify_ax(ax)
    styled_legend(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_polar_density(a1, a2, title, outpath, nbins=40):
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    h1, _ = np.histogram(a1, bins=bins, density=True)
    h2, _ = np.histogram(a2, bins=bins, density=True)

    centers_closed = np.concatenate([centers, centers[:1]])
    h1_closed = np.concatenate([h1, h1[:1]])
    h2_closed = np.concatenate([h2, h2[:1]])

    fig = plt.figure(figsize=SQUARE_FIGSIZE)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("white")
    ax.grid(color=GRID_COLOR, linestyle=":", linewidth=0.7, alpha=0.8)
    ax.plot(centers_closed, h1_closed, color=PRIMARY_COLOR, label="Modality 1")
    ax.plot(centers_closed, h2_closed, color=SECONDARY_COLOR, label="Modality 2")
    ax.set_title(title, pad=12)
    ax.set_rticks([])
    styled_legend(ax, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_joint_angle_heatmap(a1, a2, title, outpath, nbins=80):
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    hist = ax.hist2d(
        a1,
        a2,
        bins=nbins,
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
        cmap=HEATMAP_CMAP,
    )
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], linestyle="--", linewidth=0.9, color=DIAGONAL_COLOR)
    ax.set_xlabel(r"Angle $a_1$ (modality 1)")
    ax.set_ylabel(r"Angle $a_2$ (modality 2)")
    ax.set_title(title)
    prettify_ax(ax)
    cbar = fig.colorbar(hist[-1], ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Count")
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_delta_density(a1, a2, title, outpath, nbins=60):
    delta = wrap_pi_np(a2 - a1)
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    density, _ = np.histogram(delta, bins=bins, density=True)

    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    ax.plot(centers, density, color=PRIMARY_COLOR)
    ax.set_xlabel(r"$\Delta a = \mathrm{wrap}(a_2-a_1)$")
    ax.set_ylabel("Density")
    ax.set_title(title)
    prettify_ax(ax)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def animate_joint_angle_heatmap_grid_individual_cbars(
    frames_by_sigma,
    frame_steps,
    sigmas,
    outpath,
    fps=12,
    q_vmax=0.995,
    cmap=HEATMAP_CMAP,
):
    set_plot_style()

    nrows, ncols = 2, 4
    if len(sigmas) != nrows * ncols:
        raise ValueError("The animation grid expects exactly eight sigma values.")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 2.8 * nrows),
        squeeze=False,
    )
    extent = [-np.pi, np.pi, -np.pi, np.pi]

    images = []
    colorbars = []
    for idx, sigma in enumerate(sigmas):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        hist0 = frames_by_sigma[sigma][0]
        vmax0 = max(float(np.quantile(hist0, q_vmax)), 1e-6)
        image = ax.imshow(
            hist0.T,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            vmin=0.0,
            vmax=vmax0,
            cmap=cmap,
        )
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], linestyle="--", linewidth=0.9, color=DIAGONAL_COLOR)
        ax.set_title(rf"$\sigma_{{mis}}={sigma:.2f}$", pad=2)
        if col == 0:
            ax.set_ylabel(r"$a_2$")
        else:
            ax.set_yticklabels([])
        if row == nrows - 1:
            ax.set_xlabel(r"$a_1$")
        else:
            ax.set_xticklabels([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.06)
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("log(1+count)", fontsize=8, labelpad=4)
        cbar.ax.tick_params(labelsize=8)

        images.append(image)
        colorbars.append(cbar)

    suptitle = fig.suptitle(f"Cosine joint angles, step={frame_steps[0]}", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.10, top=0.90, wspace=0.25, hspace=0.30)

    def _update(frame_idx):
        suptitle.set_text(f"Cosine joint angles, step={frame_steps[frame_idx]}")
        artists = [suptitle]
        for idx, sigma in enumerate(sigmas):
            hist = frames_by_sigma[sigma][frame_idx]
            image = images[idx]
            image.set_data(hist.T)
            vmax = max(float(np.quantile(hist, q_vmax)), 1e-6)
            image.set_clim(0.0, vmax)
            colorbars[idx].update_normal(image)
            artists.append(image)
        return artists

    anim = animation.FuncAnimation(fig, _update, frames=len(frame_steps), interval=int(1000 / fps), blit=False)
    if outpath.suffix == ".gif":
        anim.save(outpath, writer=animation.PillowWriter(fps=fps))
    elif outpath.suffix == ".mp4":
        anim.save(outpath, writer=animation.FFMpegWriter(fps=fps))
    else:
        raise ValueError("Animation output must end with .gif or .mp4.")
    plt.close(fig)


def run_static_outputs(
    out_dir,
    device,
    sigmas,
    seeds,
    steps,
    B,
    lr,
    tau,
    obs_noise,
    n_eval,
    mix_w,
    kappa,
):
    gap_centers = []
    gap_band_lows = []
    gap_band_highs = []
    repr_cache = {}
    summary_label = None

    for sigma in sigmas:
        gaps = []
        for seed in seeds:
            result = run_multimodal_cosine(
                seed=seed,
                device=device,
                steps=steps,
                B=B,
                lr=lr,
                tau=tau,
                misalign_sigma=sigma,
                obs_noise=obs_noise,
                mix_w=mix_w,
                kappa=kappa,
                n_eval=n_eval,
            )
            gaps.append(result["gap"])

        summary = summarize_seed_gaps(gaps, seeds)
        gap_centers.append(summary["center"])
        gap_band_lows.append(summary["band_low"])
        gap_band_highs.append(summary["band_high"])
        summary_label = summary["label"]

        rep_result = run_multimodal_cosine(
            seed=summary["rep_seed"],
            device=device,
            steps=steps,
            B=B,
            lr=lr,
            tau=tau,
            misalign_sigma=sigma,
            obs_noise=obs_noise,
            mix_w=mix_w,
            kappa=kappa,
            n_eval=n_eval,
            return_repr=True,
        )
        repr_cache[sigma] = rep_result["angles"]

        message = (
            f"[cosine] sigma={sigma:.2f}: raw mean±std = {summary['raw_mean']:.4f} ± {summary['raw_std']:.4f}"
            f" | plotted {summary['short_label']} across {summary['n_total']} seeds"
            f" | representative seed = {summary['rep_seed']}"
        )
        print(message)

    plot_gap_curve(
        sigmas=np.array(sigmas, dtype=float),
        centers=np.array(gap_centers),
        band_lows=np.array(gap_band_lows),
        band_highs=np.array(gap_band_highs),
        title="Multimodal marginal gap vs misalignment",
        outpath=out_dir / "mm_gapcurve_cosine.pdf",
        summary_label=summary_label,
    )

    for sigma in sigmas:
        a1, a2 = repr_cache[sigma]
        sigma_tag = f"{sigma:.2f}"
        plot_polar_density(
            a1,
            a2,
            title=rf"Cosine angle density ($\sigma_{{mis}}={sigma_tag}$)",
            outpath=out_dir / f"mm_polar_cosine_sig{sigma_tag}.pdf",
        )
        plot_joint_angle_heatmap(
            a1,
            a2,
            title=rf"Cosine joint angles ($\sigma_{{mis}}={sigma_tag}$)",
            outpath=out_dir / f"mm_joint_cosine_sig{sigma_tag}.pdf",
        )
        plot_delta_density(
            a1,
            a2,
            title=rf"Cosine angle shift $\Delta a$ ($\sigma_{{mis}}={sigma_tag}$)",
            outpath=out_dir / f"mm_delta_cosine_sig{sigma_tag}.pdf",
        )


def run_animation_output(
    out_dir,
    device,
    sigmas,
    seed,
    steps,
    B,
    lr,
    tau,
    obs_noise,
    n_eval,
    mix_w,
    kappa,
    log_every,
    fps,
    animation_kind,
):
    set_seed(seed)
    theta_eval_shared = sample_theta_mixture(n_eval, w=mix_w, kappa=kappa, device=device)

    frames_by_sigma = {}
    frame_steps_ref = None
    for sigma in sigmas:
        print(f"Animating sigma={sigma:.2f} ...")
        result = run_multimodal_cosine(
            seed=seed,
            device=device,
            steps=steps,
            B=B,
            lr=lr,
            tau=tau,
            misalign_sigma=sigma,
            obs_noise=obs_noise,
            mix_w=mix_w,
            kappa=kappa,
            n_eval=n_eval,
            return_hist=True,
            nbins=80,
            log_every=log_every,
            theta_eval=theta_eval_shared,
        )
        frames_by_sigma[sigma] = result["frames"]
        if frame_steps_ref is None:
            frame_steps_ref = result["frame_steps"]
        elif frame_steps_ref != result["frame_steps"]:
            raise RuntimeError("Animation frame schedules must match across sigma values.")

    animation_path = out_dir / f"mm_joint_cosine_anim_grid_2x4_individual_cbars.{animation_kind}"
    animate_joint_angle_heatmap_grid_individual_cbars(
        frames_by_sigma=frames_by_sigma,
        frame_steps=frame_steps_ref,
        sigmas=sigmas,
        outpath=animation_path,
        fps=fps,
    )
    print(f"Saved: {animation_path.name}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run multimodal modality-gap static plots and optional animation.")
    parser.add_argument("--out-dir", default=".", help="Directory for generated figures.")
    parser.add_argument("--skip-static", action="store_true", help="Skip static PDF generation.")
    parser.add_argument("--make-animation", action="store_true", help="Also generate the joint-angle animation.")
    parser.add_argument("--animation-kind", choices=("gif", "mp4"), default="gif", help="Animation file format.")
    parser.add_argument("--animation-seed", type=int, default=0, help="Seed used for the animation run.")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of seeds for the static aggregation.")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--obs-noise", type=float, default=0.02)
    parser.add_argument("--n-eval", type=int, default=8000)
    parser.add_argument("--mix-w", type=float, default=0.7)
    parser.add_argument("--kappa", type=float, default=6.0)
    parser.add_argument("--log-every", type=int, default=10, help="Animation frame stride in optimization steps.")
    parser.add_argument("--fps", type=int, default=12, help="Animation frame rate.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    set_plot_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    sigmas = [float(sigma) for sigma in args.sigmas]
    if args.make_animation and len(sigmas) != 8:
        parser.error("The animation currently requires exactly 8 sigma values for the 2x4 grid.")

    if not args.skip_static:
        run_static_outputs(
            out_dir=out_dir,
            device=device,
            sigmas=sigmas,
            seeds=list(range(args.n_seeds)),
            steps=args.steps,
            B=args.batch_size,
            lr=args.lr,
            tau=args.tau,
            obs_noise=args.obs_noise,
            n_eval=args.n_eval,
            mix_w=args.mix_w,
            kappa=args.kappa,
        )
        print("Saved: mm_gapcurve_cosine.pdf, mm_polar_*.pdf, mm_joint_*.pdf, mm_delta_*.pdf")

    if args.make_animation:
        run_animation_output(
            out_dir=out_dir,
            device=device,
            sigmas=sigmas,
            seed=args.animation_seed,
            steps=args.steps,
            B=args.batch_size,
            lr=args.lr,
            tau=args.tau,
            obs_noise=args.obs_noise,
            n_eval=args.n_eval,
            mix_w=args.mix_w,
            kappa=args.kappa,
            log_every=args.log_every,
            fps=args.fps,
            animation_kind=args.animation_kind,
        )


if __name__ == "__main__":
    main()
