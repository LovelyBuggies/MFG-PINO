from tqdm import tqdm
import numpy as np
import torch

from .losses import PINO_loss_V, PINO_loss_rho, LpLoss
from .utils import plot_3d

try:
    import wandb
except ImportError:
    wandb = None

def eval_burgers(model,
                 dataloader,
                 v,
                 config,
                 device,
                 use_tqdm=True):
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.1)
    else:
        pbar = dataloader

    test_err = []
    f_err = []
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x).reshape(y.shape)
        out = torch.transpose(torch.squeeze(out, 0), 0, 1).cpu().detach().numpy()
        plot_3d(8, 8, out[:8, :], 'pred')
        plot_3d(8, 8, np.transpose(np.array(y[0, :, :8]), (1, 0)), 'label')
        plot_3d(8, 8, out[8:16, :], 'pred')
        plot_3d(8, 8, np.transpose(np.array(y[0, :, 8:16]), (1, 0)), 'label')
        # sio.savemat('rho.mat', {'rho': out})
        # data_loss = myloss(out, y)
        # loss_u, f_loss = PINO_loss_V(out, x[:, 0, :, 0], v)
        # test_err.append(data_loss.item())
        # f_err.append(f_loss.item())

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')