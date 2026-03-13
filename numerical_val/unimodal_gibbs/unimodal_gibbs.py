import math
import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# ----------------------------
# Plot style
# ----------------------------
def set_plot_style():
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 600,
    })


def legend_best(ax):
    leg = ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.90,
        borderpad=0.35,
        handletextpad=0,
    )
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_linewidth(0.8)
    return leg


# ----------------------------
# Repro & Math
# ----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def normalize(v, eps=1e-12):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def make_unit_np(xyz):
    x = np.asarray(xyz, dtype=np.float64)
    x = x / (np.linalg.norm(x) + 1e-12)
    return x


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


# ----------------------------
# Training & Metrics
# ----------------------------
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
):
    """
    Optimizes particles to minimize Free Energy F = U - H.
    Includes Langevin-like noise injection to prevent crystallization (bands).
    """
    set_seed(seed)
    V = torch.randn(M, 3, device=device)
    V = torch.nn.Parameter(V)
    opt = torch.optim.Adam([V], lr=lr)

    for _ in range(steps):
        Z = normalize(V)
        rho_hat = kde_on_particles_s2(Z, h=h)
        U = potential_torch_s2(Z, m1=m1, m2=m2, kappa=kappa, mix_w=mix_w)
        
        # Free Energy Loss: (Potential / tau) - Entropy
        # Note: Minimizing (U/tau + log(rho)) is equivalent to minimizing KL(rho || exp(-U/tau))
        loss = (U.mean() / tau) + torch.log(rho_hat).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        
        # --- NOISE INJECTION ---
        # Adds stochasticity to break regular lattice "bands" and emulate thermal noise
        if noise_std > 0:
            with torch.no_grad():
                V.add_(torch.randn_like(V) * noise_std)

    return normalize(V.detach())


@torch.no_grad()
def frac_in_caps_particles(Z, m1, m2, eps_rad=0.50):
    d1 = geodesic_dist_s2(Z, m1[None, :])
    d2 = geodesic_dist_s2(Z, m2[None, :])
    inside = (torch.minimum(d1, d2) <= eps_rad)
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
    n_pool=20000,
    n_draw=2500,
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


# ----------------------------
# 3D plotting
# ----------------------------
def sphere_mesh(n_u=120, n_v=60):
    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)
    return x, y, z  # each [n_v, n_u]


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


def plot_light_sphere(ax, alpha=0.15):
    x, y, z = sphere_mesh(n_u=90, n_v=45)
    ax.plot_surface(
        x, y, z,
        rstride=5, cstride=5,
        linewidth=0.5,
        edgecolor=(0.7, 0.75, 0.85, 0.4),
        antialiased=True,
        color="aliceblue",
        alpha=alpha,
        shade=True,
    )


def plot_potential_sphere(ax, m1_np, m2_np, kappa=12.0, mix_w=0.5, cmap="viridis", alpha=1.0):
    x, y, z = sphere_mesh(n_u=140, n_v=70)
    XYZ = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
    U = potential_np_s2(XYZ, m1_np, m2_np, kappa=kappa, mix_w=mix_w)
    U = U.reshape(x.shape)

    Umin, Umax = float(U.min()), float(U.max())
    norm = mcolors.Normalize(vmin=Umin, vmax=Umax)
    facecolors = plt.get_cmap(cmap)(norm(U))
    facecolors[:, :, 3] = alpha

    surf = ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )

    mappable = cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
    mappable.set_array([])

    cbar = plt.colorbar(
        mappable, 
        ax=ax, 
        location='left', 
        shrink=0.6, 
        aspect=20, 
        pad=-0.05
    )
    cbar.set_ticks([])
    cbar.set_label("Potential $U(\mathbf{z})$", rotation=90, labelpad=5)

    return surf


def draw_well_axes(ax, m1, m2):
    """Draws dashed axes from origin through m1 and m2."""
    scale = 1.2
    style_kwargs = {"linestyle": "-.", "color": "#440154", "linewidth": 0.8, "alpha":0.6 , "zorder": 100}
    ax.plot([0, m1[0]*scale], [0, m1[1]*scale], [0, m1[2]*scale], **style_kwargs)
    ax.plot([0, m2[0]*scale], [0, m2[1]*scale], [0, m2[2]*scale], **style_kwargs)


# ----------------------------
# Main experiment
# ----------------------------
def run_unimodal_training_experiment_s2(
    out_dir=".",
    device="cpu",
    taus=(0.75, 0.5, 0.25, 0.1),
    seeds=range(20),
    M=768,
    steps=5000,
    lr=5e-2,
    h=0.35,
    kappa=12.0,
    mix_w=0.5,
    eps_mass=0.50,
    gibbs_mc=120000,
    max_points_train=1400,
    gibbs_viz_pool=24000,
    gibbs_viz_draw=2600,
):
    set_plot_style()

    m1_np = make_unit_np([0.0, 0.0, 1.0])
    m2_np = make_unit_np([0.85, 0.15, -0.50])
    m1 = torch.tensor(m1_np, device=device, dtype=torch.float32)
    m2 = torch.tensor(m2_np, device=device, dtype=torch.float32)

    # Global view angle
    cam_dir = (m1_np + m2_np)
    elev, azim = view_from_direction_np(cam_dir, elev_offset=6.0, azim_offset=0.0)

    # Training and eval Loop
    mass_num_mean, mass_num_std = [], []
    mass_gib_mean, mass_gib_std = [], []
    rep_seed = list(seeds)[0]
    rep_cache = {}

    for tau in taus:
        nums = []
        for s in seeds:
            Z = train_particles_one_tau_s2(
                seed=int(s), tau=float(tau),
                device=device, M=M, steps=steps, lr=lr, h=h,
                m1=m1, m2=m2, kappa=kappa, mix_w=mix_w,
                noise_std=0.06 
            )
            nums.append(frac_in_caps_particles(Z, m1, m2, eps_rad=eps_mass))

            if int(s) == rep_seed:
                rep_cache[tau] = Z.detach().cpu().numpy()

        gibs = []
        for s in seeds:
            gibs.append(gibbs_cap_mass_mc(
                tau=float(tau), m1_np=m1_np, m2_np=m2_np, kappa=kappa, mix_w=mix_w,
                eps_rad=eps_mass, n_mc=gibbs_mc, seed=int(s),
            ))
        nums, gibs = np.array(nums), np.array(gibs)
        mass_num_mean.append(nums.mean()); mass_num_std.append(nums.std())
        mass_gib_mean.append(gibs.mean()); mass_gib_std.append(gibs.std())
        print(f"[tau={tau:g}] trained={nums.mean():.4f} | gibbs={gibs.mean():.4f}")

    # Figure 1: Spheres
    taus_list = list(taus)
    nrows, ncols = 2, 4  
    fig = plt.figure(figsize=(2.65 * ncols, 2.55 * nrows))
    gs = fig.add_gridspec(nrows, ncols, wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    # Potential sphere
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    
    plot_potential_sphere(ax0, m1_np, m2_np, kappa=kappa, mix_w=mix_w, cmap="viridis", alpha=0.8)
    
    # Draw axes through potential wells
    draw_well_axes(ax0, m1_np, m2_np)
    
    ax0.scatter([m1_np[0]], [m1_np[1]], [m1_np[2]], s=60, marker="*", color="red", alpha=1, zorder=100)
    ax0.scatter([m2_np[0]], [m2_np[1]], [m2_np[2]], s=60, marker="*", color="red", alpha=1, zorder=100)
    ax0.text(m1_np[0]*1.08, m1_np[1]*1.08, m1_np[2]*1.08+0.1, r"$\mathbf{m}_1$", fontsize=15, color="0.10", va="bottom", ha="center")
    ax0.text(m2_np[0]*1.08, m2_np[1]*1.08, m2_np[2]*1.08-0.1, r"$\mathbf{m}_2$", fontsize=15, color="0.10", va="top", ha="center")
    prettify_3d(ax0, elev=elev, azim=azim)
    ax0.text2D(0.5, 0.03, r"Potential field", transform=ax0.transAxes, ha="center", va="top", fontsize=15)

    # enumerate starts at 1, which corresponds to the flattened index of the grid
    for i, tau in enumerate(taus_list, start=1):
        # Map 1D index to 2D grid coordinates
        row = i // ncols
        col = i % ncols
        
        ax = fig.add_subplot(gs[row, col], projection="3d")
        plot_light_sphere(ax, alpha=0.15)
        
        draw_well_axes(ax, m1_np, m2_np)
        
        Xg = sample_gibbs_points_for_viz(tau=float(tau), m1_np=m1_np, m2_np=m2_np, kappa=kappa, mix_w=mix_w, n_pool=gibbs_viz_pool, n_draw=gibbs_viz_draw, seed=100 + i)
        ax.scatter(Xg[:, 0], Xg[:, 1], Xg[:, 2], s=6, alpha=0.18, marker="o", color="C0", label="Gibbs" if i == 1 else None, zorder=1)
        
        Z = rep_cache[tau]
        Zp = Z[np.random.default_rng(12345+i).choice(Z.shape[0], size=max_points_train, replace=False)] if Z.shape[0] > max_points_train else Z
        ax.scatter(Zp[:, 0], Zp[:, 1], Zp[:, 2], s=12, alpha=0.55, marker="p", color="C1", label="Trained" if i == 1 else None, zorder=10)
        
        prettify_3d(ax, elev=elev, azim=azim)
        ax.text2D(0.52, 0.06, rf"$\tau={tau:g}$", transform=ax.transAxes, ha="center", va="top", fontsize=15)
        
        # Add legend only to the first trained plot for cleanliness
        if i == 1: legend_best(ax)

    fig.savefig(f"{out_dir}/uni_train_s2_potential_and_overlays.pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)

    # Figure 2: Metrics
    taus_arr = np.array(taus_list, dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.errorbar(taus_arr, mass_gib_mean, yerr=mass_gib_std, marker="o", capsize=3, label="Gibbs")
    ax.errorbar(taus_arr, mass_num_mean, yerr=mass_num_std, marker="o", capsize=3, label="Trained")
    ax.set_xlabel(r"Temperature $\tau$")
    ax.set_ylabel(rf"Cap mass ($\varepsilon={eps_mass:g}$)")
    ax.set_title(r"Low-$\tau$ concentration (mean $\pm$ std)", fontsize=15)
    ax.set_xlim(taus_arr.max(), taus_arr.min())
    ax.grid(True, linestyle=":", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_best(ax)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/uni_train_concentration_vs_tau_s2.pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)
    
    print("Saved PDFs.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    
    run_unimodal_training_experiment_s2(
        out_dir=".",
        device=device,
        taus=(10, 5.0, 2.5, 1.0, 0.5, 0.2, 0.1),
        seeds=range(20),
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
    )