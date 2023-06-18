from argparse import ArgumentParser
import yaml

import torch
from models import FNO2d_rho, FNO2d_V
from train_utils import Adam
from train_utils.datasets import BurgersLoader
from train_utils.train_2d_couple import train_2d_couple
from train_utils.eval_2d import eval_mfg_couple


def run(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    ##
    ## Load data for Rho and V.
    ##
    dataset_rho = BurgersLoader(data_config['datapath_rho'],
                            nx=data_config['nx'], nt=data_config['nt'],
                            sub=data_config['sub'], sub_t=data_config['sub_t'], new=True)
    train_loader_rho = dataset_rho.make_loader(n_sample=data_config['n_sample'],
                                       batch_size=config['train']['batchsize'],
                                       start=data_config['offset'])

    dataset_V = BurgersLoader(data_config['datapath_V'],
                                nx=data_config['nx'] + 1, nt=data_config['nt'],
                                sub=data_config['sub'], sub_t=data_config['sub_t'], new=True)
    train_loader_V = dataset_V.make_loader(n_sample=data_config['n_sample'],
                                               batch_size=config['train']['batchsize'],
                                               start=data_config['offset'])

    model_rho = FNO2d_rho(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  act=config['model']['act']).to(device)

    model_V = FNO2d_V(modes1=config['model']['modes1'],
                      modes2=config['model']['modes2'],
                      fc_dim=config['model']['fc_dim'],
                      layers=config['model']['layers'],
                      act=config['model']['act']).to(device)

    optimizer_rho = Adam(model_rho.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    optimizer_V = Adam(model_V.parameters(), betas=(0.9, 0.999),
                         lr=config['train']['base_lr'])
    scheduler_rho = torch.optim.lr_scheduler.MultiStepLR(optimizer_rho,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    scheduler_V = torch.optim.lr_scheduler.MultiStepLR(optimizer_V,
                                                         milestones=config['train']['milestones'],
                                                         gamma=config['train']['scheduler_gamma'])
    train_2d_couple(model_rho, model_V,
                    train_loader_rho, train_loader_V,
                    optimizer_rho, optimizer_V,
                    scheduler_rho, scheduler_V,
                    config,
                    rank=0,
                    log=args.log,
                    project=config['log']['project'],
                    group=config['log']['group'])


def test(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset_rho = BurgersLoader(data_config['datapath_rho'],
                                nx=data_config['nx'], nt=data_config['nt'],
                                sub=data_config['sub'], sub_t=data_config['sub_t'], new=True)
    train_loader_rho = dataset_rho.make_loader(n_sample=data_config['n_sample'],
                                               batch_size=config['test']['batchsize'],
                                               start=data_config['offset'])

    dataset_V = BurgersLoader(data_config['datapath_V'],
                              nx=data_config['nx'] + 1, nt=data_config['nt'],
                              sub=data_config['sub'], sub_t=data_config['sub_t'], new=True)
    train_loader_V = dataset_V.make_loader(n_sample=data_config['n_sample'],
                                           batch_size=config['test']['batchsize'],
                                           start=data_config['offset'])

    model_rho = FNO2d_rho(modes1=config['model']['modes1'],
                      modes2=config['model']['modes2'],
                      fc_dim=config['model']['fc_dim'],
                      layers=config['model']['layers'],
                      act=config['model']['act']).to(device)

    model_V = FNO2d_V(modes1=config['model']['modes1'],
                    modes2=config['model']['modes2'],
                    fc_dim=config['model']['fc_dim'],
                    layers=config['model']['layers'],
                    act=config['model']['act']).to(device)

    # Load from checkpoint
    ckpt_path_rho = config['test']['ckpt_rho']
    ckpt_path_V = config['test']['ckpt_V']
    ckpt_rho = torch.load(ckpt_path_rho)
    ckpt_v = torch.load(ckpt_path_V)
    model_rho.load_state_dict(ckpt_rho['model'])
    print('Weights loaded from %s' % ckpt_path_rho)
    model_V.load_state_dict(ckpt_v['model'])
    print('Weights loaded from %s' % ckpt_path_V)
    eval_mfg_couple(model_rho, model_V,
                 train_loader_rho, train_loader_V,
                 config, device)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, help='train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(args, config)
    else:
        test(config)
