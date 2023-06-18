import torch
from tqdm import tqdm
from .utils import save_checkpoint, plot_3d
from .losses import PINO_loss_V, PINO_loss_rho, LpLoss

try:
    import wandb
except ImportError:
    wandb = None

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

            loss_u, loss_f = PINO_loss_rho(out, x[:, 0, :, 0], c)
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

        # pd.DataFrame(gap_hist).to_csv(f"diff/gap.csv")
        # pd.DataFrame(loss_hist).to_csv(f"diff/loss.csv")
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

