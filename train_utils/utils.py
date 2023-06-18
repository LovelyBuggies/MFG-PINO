import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)


def plot_3d(n_x, n_t, rho, ax_name, fig_name=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="3d")
    x = np.linspace(0, 1, n_x)
    t = np.linspace(0, 1, n_t)
    t_mesh, x_mesh = np.meshgrid(t, x)
    surf = ax.plot_surface(
        x_mesh, t_mesh, rho, cmap=cm.jet, linewidth=0, antialiased=False
    )
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=18, pad=10)

    ax.set_xlabel(r"$x$", fontsize=24, labelpad=20)
    ax.set_xlim(min(x), max(x))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    plt.ylabel(r"$t$", fontsize=24, labelpad=20)
    ax.set_ylim(min(t), max(t))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    ax.set_zlabel(ax_name, fontsize=24, labelpad=20, rotation=90)
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.view_init(elev=25, azim=-128)
    if not fig_name:
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches="tight")


def plot_diff(fig_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    rho_gap_hist = pd.read_csv(f"./diff/gap.csv")[
        "0"
    ].values.tolist()
    plt.plot(
        savgol_filter([r for r in rho_gap_hist], 33, 3),
        lw=6,
        label=r"$|\rho^{(i)} - \rho^{(i-1)}|$",
        c="indianred",
        alpha=0.8,
    )
    plt.xlabel("iterations", fontsize=24, labelpad=6)
    plt.xticks(fontsize=24)
    plt.ylabel("convergence gap", fontsize=24, labelpad=6)
    plt.yticks(fontsize=24)
    # plt.ylim(-0.01, 0.15)
    plt.legend(prop={"size": 24})
    plt.savefig(f"{fig_path}/gap.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    rho_loss_hist = pd.read_csv(f"./diff/loss.csv")[
        "0"
    ].values.tolist()
    plt.plot(
        rho_loss_hist,
        lw=6,
        label=r"$|\rho^{(i)} - \rho^*|$",
        c="steelblue",
        alpha=0.8,
    )
    plt.xlabel("iterations", fontsize=24, labelpad=6)
    plt.xticks(fontsize=24)
    plt.ylabel("loss", fontsize=24, labelpad=6)
    plt.yticks(fontsize=24)
    # plt.ylim(-0.01, 0.15)
    plt.legend(prop={"size": 24})
    plt.savefig(f"{fig_path}/loss.pdf", bbox_inches="tight")
