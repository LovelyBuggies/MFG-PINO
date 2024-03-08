# MFG-PINO

Scalable Learning for Spatiotemporal Mean-Field Games using Physics-Informed Neural Operator

```
// Put data to ./data/
python train_burgers.py --config_path configs/pretrain/nonsep_rho.yaml --mode train
python train_burgers.py --config_path configs/test/nonsep_rho.yaml --mode test
```

## Change Hyperparams

### Couple or not?

Use the rho config in `configs/pretrain` and `configs/test`;

Select make_large_loader or make_loader in `train_burgers.py`;

Change the PINO_loss in `train_2d.py`.

### Train rho or V?

Change the `config_path` argument;

Change model in `train_burgers.py`;

Change the PINO_loss in `train_2d.py`.

### Reward Function

Change the V calculation in `losses.py`.
