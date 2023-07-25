# MFG-PINO

Source code for the paper: Physics-Informed Neural Operator for Coupled Forward-Backward Partial Differential Equations.

## Change Hyperparams

### Couple or not?

Select make_large_loader or make_loader in `train_burgers.py`
Change the PINO_loss_rho's params in `train_2d.py`

### Train rho or V?

Change model in `train_burgers.py`
Change the PINO_loss_V or PINO_loss_rho in `train_2d.py`

### Reward Function

Change the V calculation in `losses.py`
