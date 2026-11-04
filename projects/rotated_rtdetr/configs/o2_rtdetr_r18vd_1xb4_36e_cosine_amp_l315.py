"""O2-RTDETR-R18 36-epoch cosine AMP baseline on L315.

This configuration keeps the L315 model, data pipeline, AdamW optimizer,
1024x1024 input, and effective batch size of eight. It uses 2000-iteration
linear warmup followed by cosine annealing over 36 epochs.

AMP is enabled with a conservative dynamic loss scale to reduce periodic
gradient overflow.
"""

from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_1xb1_72e_l315 import *


# Total training epochs.
max_epochs = 36

# Validate after every epoch so that the best checkpoint can be selected.
train_cfg.update(
    max_epochs=max_epochs,
    val_interval=1)

# Keep the original 2000-iteration linear warmup, then apply cosine annealing
# over the complete 36-epoch training schedule.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=2000),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        eta_min_ratio=0.01)
]

# Enable AMP directly in the configuration.
#
# The previous 12-epoch AMP experiment stabilized around loss scale 512.
# A long growth interval retains dynamic overflow backoff while avoiding
# repeated scale increases during this experiment.
optim_wrapper.update(
    type='AmpOptimWrapper',
    loss_scale=dict(
        init_scale=512.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100000))

# Independent output directory. This does not overwrite the 12- or 24-epoch
# experiment results.
work_dir = './work_dirs/l315_o2_rtdetr_r18_36e_cosine_amp_b4_baseline'