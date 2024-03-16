# MFG-PINO

Code for the paper: Scalable Learning for Spatiotemporal Mean-Field Games using Physics-Informed Neural Operator.

If you find it helpful, please cite us.

```
@Article{math12060803,
AUTHOR = {Liu, Shuo and Chen, Xu and Di, Xuan},
TITLE = {Scalable Learning for Spatiotemporal Mean Field Games Using Physics-Informed Neural Operator},
JOURNAL = {Mathematics},
VOLUME = {12},
YEAR = {2024},
NUMBER = {6},
ARTICLE-NUMBER = {803},
URL = {https://www.mdpi.com/2227-7390/12/6/803},
ISSN = {2227-7390},
DOI = {10.3390/math12060803}
}
```

## How to Run

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
