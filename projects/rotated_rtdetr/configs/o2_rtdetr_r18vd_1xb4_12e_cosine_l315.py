"""O2-RTDETR-R18 12-epoch cosine baseline on L315.

This configuration is intended for rapid baseline and innovation screening.
It keeps the original model, AdamW optimizer, effective batch size, dataset,
and 2000-iteration linear warmup, while replacing the constant post-warmup
learning rate with cosine annealing over 12 epochs.
"""

from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_1xb1_72e_l315 import *


max_epochs = 12

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

work_dir = './work_dirs/l315_o2_rtdetr_r18_12e_cosine_baseline'
