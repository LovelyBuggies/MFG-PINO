import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, darcy_loss, PINO_loss, PINO_loss_V, PINO_loss_rho
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None


def train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      rank=0, log=False,
                      project='PINO-2d-default',
                      group='default',
                      tags=['default'],
                      use_tqdm=True,
                      profile=False):
    '''
    train PINO on Darcy Flow
    Args:
        model:
        train_loader:
        optimizer:
        scheduler:
        config:
        rank:
        log:
        project:
        group:
        tags:
        use_tqdm:
        profile:

    Returns:

    '''
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(rank)
    pde_mesh = train_loader.dataset.pde_mesh
    pde_mol = torch.sin(np.pi * pde_mesh[..., 0]) * torch.sin(np.pi * pde_mesh[..., 1]) * 0.001
    pde_mol = pde_mol.to(rank)
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for data_ic, u, pde_ic in train_loader:
            data_ic, u, pde_ic = data_ic.to(rank), u.to(rank), pde_ic.to(rank)

            optimizer.zero_grad()

            # data loss
            if data_weight > 0:
                pred = model(data_ic).squeeze(dim=-1)
                pred = pred * mollifier
                data_loss = myloss(pred, y)

            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            loss = data_weight * data_loss + f_weight * f_loss
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * y.shape[0]
            loss_dict['f_loss'] += f_loss.item() * y.shape[0]
            loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val,
                    'data loss': data_loss_val
                }
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()
    print('Done!')


def train_2d_burger(model,
                    train_loader, v,
                    optimizer, scheduler,
                    config, c=0,
                    rank=0, log=False,
                    project='PINO-2d-default',
                    group='default',
                    tags=['default'],
                    use_tqdm=True):
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    gap_hist, loss_hist = list(), list()
    for i, e in enumerate(pbar):
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0
        train_bou = 0.0

        rho_batch, rho_label_batch = list(), list()
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)
            out = model(x).reshape(y.shape)
            rho_batch.append(out)
            rho_label_batch.append(y)

            data_loss = myloss(out, y)

            # loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0], v)
            loss_u, loss_f = PINO_loss_rho(out, x[:, 0, :, 0], c)
            # loss_u, loss_f = PINO_loss_V(out, y[:, -1, :], c)
            # loss_u, loss_f = PINO_loss_V(out, x[:, 0, :, 0], c)

            # total_loss = loss_u * ic_weight + loss_f * f_weight + data_loss * data_weight
            total_loss = loss_u * ic_weight + loss_f * f_weight
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
            train_bou += loss_u.item()

        rho = torch.stack(rho_batch)
        rho_label = torch.stack(rho_label_batch)
        if i != 0:
            gap_hist.append(float(abs(rho - tmp).mean()))
            loss_hist.append(float(abs(rho - rho_label).mean()))
        tmp = rho

        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)
        train_bou /= len(train_loader)

        pd.DataFrame(gap_hist).to_csv(f"diff/gap.csv")
        pd.DataFrame(loss_hist).to_csv(f"diff/loss.csv")
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    f'boun loss: {train_bou:.5f} '
                    f'train f error: {train_pino:.5f}; '
                    f'data l2 error: {data_l2:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': train_pino,
                    'Train L2 error': data_l2,
                    'Train loss': train_loss,
                }
            )

        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')

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