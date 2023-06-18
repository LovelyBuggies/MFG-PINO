from tqdm import tqdm
import numpy as np

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.io as sio
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from .losses import LpLoss, darcy_loss, PINO_loss

try:
    import wandb
except ImportError:
    wandb = None


def eval_darcy(model,
               dataloader,
               config,
               device,
               use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    mesh = dataloader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(device)
    f_val = []
    test_err = []

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x).reshape(y.shape)
            pred = pred * mollifier

            data_loss = myloss(pred, y)
            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            test_err.append(data_loss.item())
            f_val.append(f_loss.item())
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Equation error: {f_loss.item():.5f}, test l2 error: {data_loss.item()}'
                    )
                )
    mean_f_err = np.mean(f_val)
    std_f_err = np.std(f_val, ddof=1) / np.sqrt(len(f_val))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')


def eval_burgers(model,
                 dataloader,
                 v,
                 config,
                 device,
                 use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x).reshape(y.shape)
        rho = torch.transpose(torch.squeeze(out, 0), 0, 1).cpu().detach().numpy()
        plot_3d(8, 8, rho, 'pre')
        sio.savemat('rho.mat', {'rho': rho})
        data_loss = myloss(out, y)

        loss_u, f_loss = PINO_loss(out, x[:, 0, :, 0], v)
        test_err.append(data_loss.item())
        f_err.append(f_loss.item())

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')

def eval_mfg_couple(model_rho, model_V,
                 train_loader_rho, train_loader_V,
                 config,
                 device,
                 use_tqdm=True):
    model_rho.eval()
    model_V.eval()
    myloss = LpLoss(size_average=True)
    pbar_rho = tqdm(train_loader_rho, dynamic_ncols=True, smoothing=0.05)
    pbar_V = tqdm(train_loader_V, dynamic_ncols=True, smoothing=0.05)

    test_err_rho = []

    for x, y in pbar_rho:
        x, y = x.to(device), y.to(device)
        out = model_rho(x).reshape(y.shape)
        # plot_3d(64, 30, torch.transpose(torch.squeeze(out, 0), 0, 1).cpu().detach().numpy(), 'rho')
        plot_3d(8, 8, torch.transpose(torch.squeeze(out, 0), 0, 1).cpu().detach().numpy(), 'rho')
        data_loss = myloss(out, y)
        test_err_rho.append(data_loss.item())

    mean_err_rho = np.mean(test_err_rho)
    std_err_rho = np.std(test_err_rho, ddof=1) / np.sqrt(len(test_err_rho))

    print(f'==Averaged relative L2 error mean: {mean_err_rho}, std error: {std_err_rho}==\n')

    test_err_V = []

    for x, y in pbar_V:
        x, y = x.to(device), y.to(device)
        out = model_V(x).reshape(y.shape)
        # plot_3d(64, 30, torch.transpose(torch.squeeze(out, 0), 0, 1).cpu().detach().numpy(), 'rho')
        # plot_3d(8, 8, torch.transpose(torch.squeeze(out, 0), 0, 1).cpu().detach().numpy()[0:8, 0:8], 'V')
        data_loss = myloss(out, y)
        test_err_V.append(data_loss.item())

    mean_err_V = np.mean(test_err_V)
    std_err_V = np.std(test_err_V, ddof=1) / np.sqrt(len(test_err_V))

    print(f'==Averaged relative L2 error mean: {mean_err_V}, std error: {std_err_V}==\n')


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