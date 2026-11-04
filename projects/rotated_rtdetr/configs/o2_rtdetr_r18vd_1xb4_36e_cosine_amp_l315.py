"""O2-RTDETR-R18 在 L315 数据集上的 36 轮余弦退火 AMP 基线配置。

该配置保留 L315 模型、数据处理流程、AdamW 优化器、1024×1024 输入尺寸
以及大小为 8 的有效批量。训练开始时进行 2000 次迭代的线性预热，随后在
完整的 36 轮训练过程中使用余弦退火学习率策略。

配置采用较为保守的动态损失缩放方式启用 AMP，以减少周期性梯度溢出。
"""

from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r18vd_1xb1_72e_l315 import *


# 总训练轮数。
max_epochs = 36

# 每轮训练结束后进行一次验证，以便选出最佳模型权重。
train_cfg.update(
    max_epochs=max_epochs,
    val_interval=1)

# 保留原有的 2000 次迭代线性预热，随后在完整的 36 轮训练过程中
# 使用余弦退火学习率策略。
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

# 直接在配置文件中启用 AMP。
#
# 此前的 12 轮 AMP 实验表明，损失缩放值在 512 左右趋于稳定。
# 较长的增长间隔既保留了发生溢出时的动态回退机制，也可避免本次实验中
# 损失缩放值被反复提高。
optim_wrapper.update(
    type='AmpOptimWrapper',
    loss_scale=dict(
        init_scale=512.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100000))

# 使用独立的输出目录，不会覆盖 12 轮或 24 轮实验结果。
work_dir = './work_dirs/l315_o2_rtdetr_r18_36e_cosine_amp_b4_baseline'
