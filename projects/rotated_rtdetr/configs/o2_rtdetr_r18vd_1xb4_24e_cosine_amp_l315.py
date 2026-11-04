"""O2-RTDETR-R18 24-epoch cosine AMP baseline on L315.

This configuration keeps the L315 model, data pipeline, AdamW optimizer,
1024x1024 input, and effective batch size of eight.  It extends cosine
annealing to 24 epochs and enables AMP with a conservative dynamic loss scale
to avoid the periodic gradient overflows observed with the default scaler.
"""

from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_1xb1_72e_l315 import *


max_epochs = 24

train_cfg.update(
    max_epochs=max_epochs,
    val_interval=1)

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

# The 12-epoch AMP run repeatedly grew beyond its stable scale and then
# backed off to 512. Keep dynamic backoff enabled, but prevent unnecessary
# scale growth during this 24-epoch experiment.
optim_wrapper.update(
    type='AmpOptimWrapper',
    loss_scale=dict(
        init_scale=512.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100000))

work_dir = './work_dirs/l315_o2_rtdetr_r18_24e_cosine_amp_b4_baseline'
