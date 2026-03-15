import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation


NUMERICAL_ROOT = Path(__file__).resolve().parents[1]
if str(NUMERICAL_ROOT) not in sys.path:
    sys.path.append(str(NUMERICAL_ROOT))

from plot_style import (
    ACCENT_COLOR,
    CURVE_FIGSIZE,
    GRID_COLOR,
    HIGHLIGHT_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SURFACE_CMAP,
    THREE_D_PANEL_SIZE,
    grid_figure_size,
    prettify_ax,
    set_plot_style,
    styled_legend,
)


DEFAULT_METRIC_TAUS = (10.0, 5.0, 2.5, 1.0, 0.5, 0.2, 0.1)


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    taus: tuple[float, ...]
    nrows: int
    ncols: int
    filename: str


LAYOUTS = {
    "grid2x4": LayoutSpec(
        name="grid2x4",
        taus=DEFAULT_METRIC_TAUS,
        nrows=2,
        ncols=4,
        filename="uni_train_s2_potential_and_overlays.pdf",
    ),
    "row5": LayoutSpec(
        name="row5",
        taus=(10.0, 2.5, 0.5, 0.1),
        nrows=1,
        ncols=5,
        filename="uni_train_s2_overlay_row5.pdf",
    ),
    "row4": LayoutSpec(
        name="row4",
        taus=(10.0, 1.0, 0.1),
        nrows=1,
        ncols=4,
        filename="uni_train_s2_overlay_row4.pdf",
    ),
}


@dataclass(frozen=True)
class SphereSetup:
    m1_np: np.ndarray
    m2_np: np.ndarray
    m1: torch.Tensor
    m2: torch.Tensor
    elev: float
    azim: float


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def normalize(v, eps=1e-12):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def make_unit_np(xyz):
    x = np.asarray(xyz, dtype=np.float64)
    return x / (np.linalg.norm(x) + 1e-12)


def pairwise_geodesic_dist_s2(Z, eps=1e-6):
    dots = (Z @ Z.t()).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(dots)


def geodesic_dist_s2(z, w, eps=1e-6):
    dot = (z * w).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(dot)


def potential_torch_s2(Z, m1, m2, kappa=12.0, mix_w=0.5):
    s1 = (Z * m1).sum(dim=-1)
    s2 = (Z * m2).sum(dim=-1)
    a = kappa * s1 + math.log(max(mix_w, 1e-12))
    b = kappa * s2 + math.log(max(1.0 - mix_w, 1e-12))
    mx = torch.maximum(a, b)
    lse = mx + torch.log(torch.exp(a - mx) + torch.exp(b - mx))
    return -(1.0 / kappa) * lse


def potential_np_s2(Z, m1, m2, kappa=12.0, mix_w=0.5):
    s1 = (Z * m1[None, :]).sum(axis=1)
    s2 = (Z * m2[None, :]).sum(axis=1)
    a = kappa * s1 + math.log(max(mix_w, 1e-12))
    b = kappa * s2 + math.log(max(1.0 - mix_w, 1e-12))
    mx = np.maximum(a, b)
    lse = mx + np.log(np.exp(a - mx) + np.exp(b - mx))
    return -(1.0 / kappa) * lse


def kde_on_particles_s2(Z, h, eps=1e-12):
    D = pairwise_geodesic_dist_s2(Z)
    K = torch.exp(-(D * D) / (2.0 * h * h))
    return K.mean(dim=1) + eps


def train_particles_one_tau_s2(
    seed: int,
    tau: float,
    device="cpu",
    M: int = 768,
    steps: int = 5000,
    lr: float = 5e-2,
    h: float = 0.35,
    m1=None,
    m2=None,
    kappa=12.0,
    mix_w=0.5,
    noise_std: float = 0.06,
    return_history: bool = False,
    log_every: int = 50,
):
    set_seed(seed)
    V = torch.randn(M, 3, device=device)
    V = torch.nn.Parameter(V)
    opt = torch.optim.Adam([V], lr=lr)

    history = [] if return_history else None
    # Keep the CUDA/CPU execution path aligned whether or not we save animation history.
    initial_snapshot = normalize(V.detach()).cpu().numpy()
    if return_history:
        history.append((0, initial_snapshot))

    for step in range(steps):
        Z = normalize(V)
        rho_hat = kde_on_particles_s2(Z, h=h)
        U = potential_torch_s2(Z, m1=m1, m2=m2, kappa=kappa, mix_w=mix_w)
        loss = (U.mean() / tau) + torch.log(rho_hat).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if noise_std > 0:
            with torch.no_grad():
                V.add_(torch.randn_like(V) * noise_std)

        step_now = step + 1
        if step_now % log_every == 0 or step_now == steps:
            snapshot = normalize(V.detach()).cpu().numpy()
            if return_history:
                history.append((step_now, snapshot))

    Z_final = normalize(V.detach())
    if return_history:
        return Z_final, history
    return Z_final


@torch.no_grad()
def frac_in_caps_particles(Z, m1, m2, eps_rad=0.50):
    d1 = geodesic_dist_s2(Z, m1[None, :])
    d2 = geodesic_dist_s2(Z, m2[None, :])
    inside = torch.minimum(d1, d2) <= eps_rad
    return float(inside.float().mean().cpu())


def gibbs_cap_mass_mc(
    tau,
    m1_np,
    m2_np,
    kappa=12.0,
    mix_w=0.5,
    eps_rad=0.50,
    n_mc=120000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_mc, 3))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    U = potential_np_s2(X, m1_np, m2_np, kappa=kappa, mix_w=mix_w)
    logw = -U / tau
    logw = logw - np.max(logw)
    w = np.exp(logw)

    dot1 = np.clip((X * m1_np[None, :]).sum(axis=1), -1.0, 1.0)
    dot2 = np.clip((X * m2_np[None, :]).sum(axis=1), -1.0, 1.0)
    d1 = np.arccos(dot1)
    d2 = np.arccos(dot2)
    inside = (np.minimum(d1, d2) <= eps_rad).astype(np.float64)
    return float((w * inside).sum() / (w.sum() + 1e-12))


def sample_gibbs_points_for_viz(
    tau,
    m1_np,
    m2_np,
    kappa=12.0,
    mix_w=0.5,
    n_pool=24000,
    n_draw=2400,
    seed=0,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_pool, 3))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    U = potential_np_s2(X, m1_np, m2_np, kappa=kappa, mix_w=mix_w)
    logw = -U / tau
    logw = logw - np.max(logw)
    w = np.exp(logw)
    p = w / (w.sum() + 1e-12)

    idx = rng.choice(n_pool, size=min(n_draw, n_pool), replace=False, p=p)
    return X[idx]


def sphere_mesh(n_u=120, n_v=60):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    return x, y, z


def view_from_direction_np(dir_vec, elev_offset=0.0, azim_offset=0.0):
    d = np.asarray(dir_vec, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    elev = np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))) + elev_offset
    azim = np.degrees(np.arctan2(d[1], d[0])) + azim_offset
    return float(elev), float(azim)


def prettify_3d(ax, elev=18, azim=35):
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_facecolor("white")


def plot_light_sphere(ax, alpha=0.16):
    x, y, z = sphere_mesh(n_u=90, n_v=45)
    ax.plot_surface(
        x,
        y,
        z,
        rstride=5,
        cstride=5,
        linewidth=0.5,
        edgecolor=(*mcolors.to_rgb(GRID_COLOR), 0.45),
        antialiased=True,
        color="aliceblue",
        alpha=alpha,
        shade=True,
    )


def plot_potential_sphere(ax, m1_np, m2_np, kappa=12.0, mix_w=0.5, cmap=SURFACE_CMAP, alpha=0.85):
    x, y, z = sphere_mesh(n_u=140, n_v=70)
    XYZ = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
    U = potential_np_s2(XYZ, m1_np, m2_np, kappa=kappa, mix_w=mix_w).reshape(x.shape)

    norm = mcolors.Normalize(vmin=float(U.min()), vmax=float(U.max()))
    facecolors = plt.get_cmap(cmap)(norm(U))
    facecolors[:, :, 3] = alpha
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )

    mappable = cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    mappable.set_array([])
    return mappable


def precompute_potential_surface_s2(m1_np, m2_np, kappa=12.0, mix_w=0.5, cmap=SURFACE_CMAP, n_u=140, n_v=70):
    x, y, z = sphere_mesh(n_u=n_u, n_v=n_v)
    XYZ = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
    U = potential_np_s2(XYZ, m1_np, m2_np, kappa=kappa, mix_w=mix_w).reshape(x.shape)

    norm = mcolors.Normalize(vmin=float(U.min()), vmax=float(U.max()))
    cmap_obj = plt.get_cmap(cmap)
    return x, y, z, U, norm, cmap_obj


def plot_potential_surface_precomputed(ax, x, y, z, U, norm, cmap_obj, alpha=0.35):
    facecolors = cmap_obj(norm(U))
    facecolors[:, :, 3] = alpha
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )


def draw_well_axes(ax, m1, m2):
    scale = 1.2
    style_kwargs = {
        "linestyle": "-.",
        "color": ACCENT_COLOR,
        "linewidth": 0.9,
        "alpha": 0.65,
        "zorder": 100,
    }
    ax.plot([0, m1[0] * scale], [0, m1[1] * scale], [0, m1[2] * scale], **style_kwargs)
    ax.plot([0, m2[0] * scale], [0, m2[1] * scale], [0, m2[2] * scale], **style_kwargs)


def add_well_markers(ax, setup):
    ax.scatter([setup.m1_np[0]], [setup.m1_np[1]], [setup.m1_np[2]], s=60, marker="*", color=HIGHLIGHT_COLOR, zorder=120)
    ax.scatter([setup.m2_np[0]], [setup.m2_np[1]], [setup.m2_np[2]], s=60, marker="*", color=HIGHLIGHT_COLOR, zorder=120)
    ax.text(
        setup.m1_np[0] * 1.08,
        setup.m1_np[1] * 1.08,
        setup.m1_np[2] * 1.08 + 0.1,
        r"$\mathbf{m}_1$",
        fontsize=11,
        color="0.12",
        va="bottom",
        ha="center",
    )
    ax.text(
        setup.m2_np[0] * 1.08,
        setup.m2_np[1] * 1.08,
        setup.m2_np[2] * 1.08 - 0.1,
        r"$\mathbf{m}_2$",
        fontsize=11,
        color="0.12",
        va="top",
        ha="center",
    )


def subsample_points(points, max_points, seed):
    if max_points is None or points.shape[0] <= max_points:
        return points
    idx = np.random.default_rng(seed).choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def build_sphere_setup(device):
    m1_np = make_unit_np([0.0, 0.0, 1.0])
    m2_np = make_unit_np([0.85, 0.15, -0.50])
    cam_dir = m1_np + m2_np
    elev, azim = view_from_direction_np(cam_dir, elev_offset=6.0, azim_offset=0.0)
    return SphereSetup(
        m1_np=m1_np,
        m2_np=m2_np,
        m1=torch.tensor(m1_np, device=device, dtype=torch.float32),
        m2=torch.tensor(m2_np, device=device, dtype=torch.float32),
        elev=elev,
        azim=azim,
    )


def create_layout_figure(layout):
    fig = plt.figure(figsize=grid_figure_size(layout.nrows, layout.ncols, panel_size=THREE_D_PANEL_SIZE))
    gs = fig.add_gridspec(
        layout.nrows,
        layout.ncols,
        wspace=0.02,
        hspace=0.02,
        left=0.02,
        right=0.995,
        top=0.98,
        bottom=0.03,
    )
    return fig, gs


def add_reference_panel(fig, gs, layout, setup, kappa, mix_w, cmap=SURFACE_CMAP):
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    mappable = plot_potential_sphere(ax0, setup.m1_np, setup.m2_np, kappa=kappa, mix_w=mix_w, cmap=cmap, alpha=0.85)
    draw_well_axes(ax0, setup.m1_np, setup.m2_np)
    add_well_markers(ax0, setup)
    prettify_3d(ax0, elev=setup.elev, azim=setup.azim)
    ax0.text2D(0.5, 0.04, r"Potential field", transform=ax0.transAxes, ha="center", va="top", fontsize=11)

    cbar = fig.colorbar(
        mappable,
        ax=ax0,
        location="left",
        shrink=0.62,
        aspect=22,
        pad=-0.03,
    )
    cbar.set_ticks([])
    cbar.set_label(r"Potential $U(\mathbf{z})$", rotation=90, labelpad=6)
    return ax0


def add_static_tau_panel(
    fig,
    gs,
    layout,
    panel_index,
    tau,
    setup,
    rep_cache,
    max_points_train,
    gibbs_viz_pool,
    gibbs_viz_draw,
    kappa,
    mix_w,
):
    row = panel_index // layout.ncols
    col = panel_index % layout.ncols
    ax = fig.add_subplot(gs[row, col], projection="3d")
    plot_light_sphere(ax, alpha=0.16)
    draw_well_axes(ax, setup.m1_np, setup.m2_np)

    Xg = sample_gibbs_points_for_viz(
        tau=float(tau),
        m1_np=setup.m1_np,
        m2_np=setup.m2_np,
        kappa=kappa,
        mix_w=mix_w,
        n_pool=gibbs_viz_pool,
        n_draw=gibbs_viz_draw,
        seed=100 + panel_index,
    )
    ax.scatter(
        Xg[:, 0],
        Xg[:, 1],
        Xg[:, 2],
        s=6,
        alpha=0.18,
        marker="o",
        color=PRIMARY_COLOR,
        label="Gibbs" if panel_index == 1 else None,
        zorder=1,
    )

    Zp = subsample_points(rep_cache[tau], max_points_train, seed=12345 + panel_index)
    ax.scatter(
        Zp[:, 0],
        Zp[:, 1],
        Zp[:, 2],
        s=12,
        alpha=0.55,
        marker="p",
        color=SECONDARY_COLOR,
        label="Trained" if panel_index == 1 else None,
        zorder=10,
    )

    prettify_3d(ax, elev=setup.elev, azim=setup.azim)
    ax.text2D(0.5, 0.04, rf"$\tau={tau:g}$", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    if panel_index == 1:
        styled_legend(ax, loc="upper left", ncol=2)
    return ax


def save_static_layout(
    out_dir,
    layout,
    setup,
    rep_cache,
    max_points_train,
    gibbs_viz_pool,
    gibbs_viz_draw,
    kappa,
    mix_w,
    cmap=SURFACE_CMAP,
):
    fig, gs = create_layout_figure(layout)
    add_reference_panel(fig, gs, layout, setup, kappa=kappa, mix_w=mix_w, cmap=cmap)
    for panel_index, tau in enumerate(layout.taus, start=1):
        add_static_tau_panel(
            fig=fig,
            gs=gs,
            layout=layout,
            panel_index=panel_index,
            tau=tau,
            setup=setup,
            rep_cache=rep_cache,
            max_points_train=max_points_train,
            gibbs_viz_pool=gibbs_viz_pool,
            gibbs_viz_draw=gibbs_viz_draw,
            kappa=kappa,
            mix_w=mix_w,
        )

    out_path = out_dir / layout.filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def animate_layout(
    out_path,
    layout,
    rep_hist,
    setup,
    kappa,
    mix_w,
    max_points_train=2000,
    fps=12,
    writer_kind="gif",
    overlay_alpha=0.35,
    cmap=SURFACE_CMAP,
):
    fig, gs = create_layout_figure(layout)
    add_reference_panel(fig, gs, layout, setup, kappa=kappa, mix_w=mix_w, cmap=cmap)

    x, y, z, Umesh, Unorm, cmap_obj = precompute_potential_surface_s2(
        m1_np=setup.m1_np,
        m2_np=setup.m2_np,
        kappa=kappa,
        mix_w=mix_w,
        cmap=cmap,
    )

    train_scatters = {}
    step_texts = {}
    train_indices = {}
    n_frames = min(len(rep_hist[tau]) for tau in layout.taus)

    for panel_index, tau in enumerate(layout.taus, start=1):
        row = panel_index // layout.ncols
        col = panel_index % layout.ncols
        ax = fig.add_subplot(gs[row, col], projection="3d")
        plot_potential_surface_precomputed(ax, x, y, z, Umesh, Unorm, cmap_obj, alpha=overlay_alpha)
        draw_well_axes(ax, setup.m1_np, setup.m2_np)

        step0, Z0 = rep_hist[tau][0]
        idx = np.arange(Z0.shape[0])
        if max_points_train is not None and max_points_train < Z0.shape[0]:
            idx = np.random.default_rng(12345 + panel_index).choice(Z0.shape[0], size=max_points_train, replace=False)
        train_indices[tau] = idx

        Z0p = Z0[idx]
        scatter = ax.scatter(
            Z0p[:, 0],
            Z0p[:, 1],
            Z0p[:, 2],
            s=12,
            alpha=0.55,
            marker="p",
            color=SECONDARY_COLOR,
            label="Trained" if panel_index == 1 else None,
            zorder=10,
            depthshade=False,
        )
        train_scatters[tau] = scatter

        prettify_3d(ax, elev=setup.elev, azim=setup.azim)
        ax.text2D(0.5, 0.05, rf"$\tau={tau:g}$", transform=ax.transAxes, ha="center", va="top", fontsize=11)
        step_text = ax.text2D(0.5, 0.00, f"step={step0}", transform=ax.transAxes, ha="center", va="top", fontsize=9, color="0.30")
        step_texts[tau] = step_text
        if panel_index == 1:
            styled_legend(ax, loc="upper left")

    def _update(frame_idx):
        artists = []
        for tau in layout.taus:
            step, Z = rep_hist[tau][frame_idx]
            Zp = Z[train_indices[tau]]
            scatter = train_scatters[tau]
            scatter._offsets3d = (Zp[:, 0], Zp[:, 1], Zp[:, 2])
            step_texts[tau].set_text(f"step={step}")
            artists.extend([scatter, step_texts[tau]])
        return artists

    anim = animation.FuncAnimation(fig, _update, frames=n_frames, interval=int(1000 / fps), blit=False)
    if writer_kind == "gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    elif writer_kind == "mp4":
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps))
    else:
        raise ValueError("writer_kind must be 'gif' or 'mp4'.")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_mass_curve(out_dir, taus, mass_num_mean, mass_num_std, mass_gib_mean, mass_gib_std, eps_mass):
    taus_arr = np.array(taus, dtype=float)
    fig, ax = plt.subplots(figsize=CURVE_FIGSIZE)
    ax.errorbar(taus_arr, mass_gib_mean, yerr=mass_gib_std, marker="o", capsize=3, color=PRIMARY_COLOR, label="Gibbs")
    ax.errorbar(taus_arr, mass_num_mean, yerr=mass_num_std, marker="o", capsize=3, color=SECONDARY_COLOR, label="Trained")
    ax.set_xlabel(r"Temperature $\tau$")
    ax.set_ylabel(rf"Cap mass ($\varepsilon={eps_mass:g}$)")
    ax.set_title(r"Low-$\tau$ concentration (mean $\pm$ std)")
    ax.set_xlim(taus_arr.max(), taus_arr.min())
    prettify_ax(ax)
    styled_legend(ax, loc="upper left")
    fig.tight_layout()
    out_path = out_dir / "uni_train_concentration_vs_tau_s2.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def run_unimodal_training_experiment_s2(
    out_dir=".",
    device="cpu",
    taus=DEFAULT_METRIC_TAUS,
    layout_names=("grid2x4", "row5", "row4"),
    seeds=range(20),
    rep_seed=0,
    M=768,
    steps=5000,
    lr=5e-2,
    h=0.35,
    kappa=12.0,
    mix_w=0.5,
    eps_mass=0.50,
    gibbs_mc=120000,
    max_points_train=2000,
    gibbs_viz_pool=24000,
    gibbs_viz_draw=2400,
    make_animation=False,
    animation_layout="grid2x4",
    log_every=50,
    anim_fps=12,
    anim_kind="gif",
    overlay_alpha=0.35,
    cmap=SURFACE_CMAP,
    skip_static=False,
    skip_metrics=False,
):
    set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_taus = tuple(float(tau) for tau in taus)
    selected_layouts = [LAYOUTS[name] for name in layout_names]
    anim_layout = LAYOUTS[animation_layout]
    required_taus = list(metric_taus)
    for layout in selected_layouts:
        required_taus.extend(layout.taus)
    if make_animation:
        required_taus.extend(anim_layout.taus)
    all_taus = tuple(dict.fromkeys(float(tau) for tau in required_taus))

    setup = build_sphere_setup(device)
    seeds = [int(seed) for seed in seeds]
    rep_seed = int(rep_seed)

    mass_num_mean = {}
    mass_num_std = {}
    mass_gib_mean = {}
    mass_gib_std = {}
    rep_cache = {}
    rep_hist = {}

    for tau in all_taus:
        nums = []
        need_history = make_animation and tau in anim_layout.taus
        for seed in seeds:
            if seed == rep_seed and need_history:
                Z_final, history = train_particles_one_tau_s2(
                    seed=seed,
                    tau=tau,
                    device=device,
                    M=M,
                    steps=steps,
                    lr=lr,
                    h=h,
                    m1=setup.m1,
                    m2=setup.m2,
                    kappa=kappa,
                    mix_w=mix_w,
                    noise_std=0.06,
                    return_history=True,
                    log_every=log_every,
                )
                rep_cache[tau] = Z_final.detach().cpu().numpy()
                rep_hist[tau] = history
                Z = Z_final
            else:
                Z = train_particles_one_tau_s2(
                    seed=seed,
                    tau=tau,
                    device=device,
                    M=M,
                    steps=steps,
                    lr=lr,
                    h=h,
                    m1=setup.m1,
                    m2=setup.m2,
                    kappa=kappa,
                    mix_w=mix_w,
                    noise_std=0.06,
                    return_history=False,
                )
                if seed == rep_seed:
                    rep_cache[tau] = Z.detach().cpu().numpy()

            if tau in metric_taus:
                nums.append(frac_in_caps_particles(Z, setup.m1, setup.m2, eps_rad=eps_mass))

        if tau in metric_taus:
            gibs = [
                gibbs_cap_mass_mc(
                    tau=tau,
                    m1_np=setup.m1_np,
                    m2_np=setup.m2_np,
                    kappa=kappa,
                    mix_w=mix_w,
                    eps_rad=eps_mass,
                    n_mc=gibbs_mc,
                    seed=seed,
                )
                for seed in seeds
            ]
            nums = np.array(nums)
            gibs = np.array(gibs)
            mass_num_mean[tau] = float(nums.mean())
            mass_num_std[tau] = float(nums.std())
            mass_gib_mean[tau] = float(gibs.mean())
            mass_gib_std[tau] = float(gibs.std())
            print(f"[tau={tau:g}] trained={nums.mean():.4f} | gibbs={gibs.mean():.4f}")

    if not skip_static:
        for layout in selected_layouts:
            save_static_layout(
                out_dir=out_dir,
                layout=layout,
                setup=setup,
                rep_cache=rep_cache,
                max_points_train=max_points_train,
                gibbs_viz_pool=gibbs_viz_pool,
                gibbs_viz_draw=gibbs_viz_draw,
                kappa=kappa,
                mix_w=mix_w,
                cmap=cmap,
            )

    if not skip_metrics:
        plot_mass_curve(
            out_dir=out_dir,
            taus=metric_taus,
            mass_num_mean=np.array([mass_num_mean[tau] for tau in metric_taus]),
            mass_num_std=np.array([mass_num_std[tau] for tau in metric_taus]),
            mass_gib_mean=np.array([mass_gib_mean[tau] for tau in metric_taus]),
            mass_gib_std=np.array([mass_gib_std[tau] for tau in metric_taus]),
            eps_mass=eps_mass,
        )

    if make_animation:
        animation_path = out_dir / f"uni_train_s2_{anim_layout.name}_evolution.{anim_kind}"
        animate_layout(
            out_path=animation_path,
            layout=anim_layout,
            rep_hist=rep_hist,
            setup=setup,
            kappa=kappa,
            mix_w=mix_w,
            max_points_train=max_points_train,
            fps=anim_fps,
            writer_kind=anim_kind,
            overlay_alpha=overlay_alpha,
            cmap=cmap,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Run unimodal Gibbs sphere layouts, metrics, and optional animations.")
    parser.add_argument("--out-dir", default=".", help="Directory for generated figures.")
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_METRIC_TAUS), help="Temperatures for the metric curve.")
    parser.add_argument("--layouts", nargs="+", choices=tuple(LAYOUTS.keys()), default=["grid2x4", "row5", "row4"])
    parser.add_argument("--skip-static", action="store_true", help="Skip static sphere layout PDFs.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip the concentration-vs-temperature plot.")
    parser.add_argument("--make-animation", action="store_true", help="Also save an animation from the representative seed.")
    parser.add_argument("--animation-layout", choices=tuple(LAYOUTS.keys()), default="grid2x4")
    parser.add_argument("--animation-kind", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--rep-seed", type=int, default=0, help="Seed used for representative static and animated panels.")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of seeds for metric aggregation.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--num-particles", type=int, default=768)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--bandwidth", type=float, default=0.35)
    parser.add_argument("--kappa", type=float, default=12.0)
    parser.add_argument("--mix-w", type=float, default=0.5)
    parser.add_argument("--eps-mass", type=float, default=0.50)
    parser.add_argument("--gibbs-mc", type=int, default=120000)
    parser.add_argument("--max-points-train", type=int, default=2000)
    parser.add_argument("--gibbs-viz-pool", type=int, default=24000)
    parser.add_argument("--gibbs-viz-draw", type=int, default=2400)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    set_plot_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    run_unimodal_training_experiment_s2(
        out_dir=args.out_dir,
        device=device,
        taus=tuple(args.taus),
        layout_names=tuple(args.layouts),
        seeds=range(args.n_seeds),
        rep_seed=args.rep_seed,
        M=args.num_particles,
        steps=args.steps,
        lr=args.lr,
        h=args.bandwidth,
        kappa=args.kappa,
        mix_w=args.mix_w,
        eps_mass=args.eps_mass,
        gibbs_mc=args.gibbs_mc,
        max_points_train=args.max_points_train,
        gibbs_viz_pool=args.gibbs_viz_pool,
        gibbs_viz_draw=args.gibbs_viz_draw,
        make_animation=args.make_animation,
        animation_layout=args.animation_layout,
        log_every=args.log_every,
        anim_fps=args.fps,
        anim_kind=args.animation_kind,
        overlay_alpha=args.overlay_alpha,
        skip_static=args.skip_static,
        skip_metrics=args.skip_metrics,
    )


if __name__ == "__main__":
    main()
