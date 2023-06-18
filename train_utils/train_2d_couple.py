import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, PINO_loss_rho, PINO_loss_V
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def train_2d_couple(model_rho, model_V,
                    train_loader_rho, train_loader_V,
                    optimizer_rho, optimizer_V,
                    scheduler_rho, scheduler_V,
                    config, c=0,
                    rank=0, log=False,
                    project='PINO-2d-default',
                    group='default',
                    tags=['default'],
                    use_tqdm=True):


    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    model_rho.train()
    model_V.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    u = torch.full((config['data']['n_sample'], config['data']['nx'], config['data']['nt'] + 1), 0.5)
    # u_hist = list()
    for e in pbar:
        model_rho.train()
        model_V.train()
        for _ in range(config['train']['step_rho']):
            train_pino = 0.0
            data_l2 = 0.0
            train_loss = 0.0
            train_bou = 0.0
            rho_batch = list()
            for rhoi, (x, y) in enumerate(train_loader_rho):
                x, y = x.to(rank), y.to(rank)
                out = model_rho(x).reshape(y.shape)
                data_loss = myloss(out, y)

                loss_u, loss_f = PINO_loss_rho(out, x[:, 0, :, 0], u)
                total_loss_rho = loss_u * ic_weight + loss_f * f_weight
                optimizer_rho.zero_grad()
                total_loss_rho.backward(retain_graph=True)
                optimizer_rho.step()

                data_l2 += data_loss.item()
                train_pino += loss_f.item()
                train_loss += total_loss_rho.item()
                train_bou += loss_u.item()

                rho_i = model_rho(x).reshape(y.shape)
                rho_batch.append(rho_i)

            rho = torch.stack(rho_batch).squeeze(0).detach()
            scheduler_rho.step()
            data_l2 /= len(train_loader_rho)
            train_pino /= len(train_loader_rho)
            train_loss /= len(train_loader_rho)
            train_bou /= len(train_loader_rho)
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Epoch {e}, train loss: {train_loss:.5f} '
                        f'boun loss: {train_bou:.5f} '
                        f'train f error: {train_pino:.5f}; '
                        f'data l2 rho error: {data_l2:.5f}'
                    )
                )

        for _ in range(config['train']['step_V']):
            train_pino = 0.0
            data_l2 = 0.0
            train_loss = 0.0
            train_bou = 0.0
            u_batch = list()
            for Vi, (x, y) in enumerate(train_loader_V):
                x, y = x.to(rank), y.to(rank)
                out_V = model_V(x).reshape(y.shape)
                data_loss_V = myloss(out_V, y)

                loss_u_V, loss_f_V = PINO_loss_V(out_V, x[:, 0, :, 0], rho)
                total_loss_V = loss_u_V * ic_weight + loss_f_V * f_weight
                optimizer_V.zero_grad()
                total_loss_V.backward(retain_graph=True)
                optimizer_V.step()

                data_l2 += data_loss_V.item()
                train_pino += loss_f.item()
                train_loss += total_loss_V.item()
                train_bou += loss_u.item()

                # V = model_V(x).reshape(y.shape)
                V = y
                delta_t = 1/ 8
                for t in range(7):
                    for i in range(8):
                        speed = min(max((V[Vi, t + 1, i] - V[Vi, t + 1, i + 1]) / delta_t + 1 - rho[Vi, t, i], 0), 1)
                        u[Vi, t, i] = speed
                        V[Vi, t, i] = delta_t * (0.5 * speed ** 2 + rho[Vi, t, i] * speed - speed) + \
                                      (1 - speed) * V[Vi, t + 1, i] + speed * V[Vi, t + 1, i + 1]

                V_x = (V[:, :, 1:] - V[:, :, :-1]) / (1 / config['data']['nx'])
                u_i = (torch.ones((config['data']['n_sample'], 8, 8), dtype=torch.float32).to('cuda:0') - rho - V_x.to('cuda:0')).detach()
                # V = torch.zeros((config['data']['n_sample'], 9, 9), dtype=torch.float32).to('cuda:0')
                # u = torch.zeros((config['data']['n_sample'], 8, 8), dtype=torch.float32).to('cuda:0')
                # delta_t = 1 / 8
                # for t in range(7, -1, -1):
                #     for i in range(8):
                #         speed = min(max((V[Vi, t + 1, i] - V[Vi, t + 1, i + 1]) / delta_t + 1 - rho[Vi, t, i], 0), 1)
                #         u[Vi, t, i] = speed
                #         V[Vi, t, i] = delta_t * (0.5 * speed ** 2 + rho[Vi, t, i] * speed - speed) + \
                #                       (1 - speed) * V[Vi, t + 1, i] + speed * V[Vi, t + 1, i + 1]
                #
                #     V[Vi, t, 8] = V[Vi, t, 0]

                u_batch.append(u_i)

            u = torch.stack(u_batch).squeeze(0).detach()
            scheduler_V.step()
            data_l2 /= len(train_loader_V)
            train_pino /= len(train_loader_V)
            train_loss /= len(train_loader_V)
            train_bou /= len(train_loader_V)
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Epoch {e}, train loss: {train_loss:.5f} '
                        f'boun loss: {train_bou:.5f} '
                        f'train f error: {train_pino:.5f}; '
                        f'data l2 V error: {data_l2:.5f}'
                    )
                )

        # u_hist.append(u)
        # u = torch.mean(torch.stack(u_hist, dim=0), dim=0)
        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir_rho'],
                            config['train']['save_name_rho'].replace('.pt', f'_{e}.pt'),
                            model_rho, optimizer_rho)
            save_checkpoint(config['train']['save_dir_V'],
                            config['train']['save_name_V'].replace('.pt', f'_{e}.pt'),
                            model_V, optimizer_V)
    save_checkpoint(config['train']['save_dir_rho'],
                    config['train']['save_name_rho'],
                    model_rho, optimizer_rho)
    save_checkpoint(config['train']['save_dir_V'],
                    config['train']['save_name_V'],
                    model_V, optimizer_V)
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